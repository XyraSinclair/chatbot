

import { Matrix as MlMatrix } from "npm:ml-matrix"; // https://github.com/mljs/matrix
import { dequal } from "npm:dequal";
import { unityNormalize } from "../array/normalize.ts";
import { geometricMean, sum } from "../array/unary.ts";
import { zip } from "../array/zip.ts";
import { Letters } from "../letters.ts";
import { Index } from "../types/type_aliases.ts";
import { ArrMatrix } from "./matrix.ts";
import { MatrixT, Square } from "./type_matrix.ts";
import { ReadonlyDeep, WritableDeep } from "../types/type_readonly.ts";

/**
 * Positive real number
 */
type PosReal = number;
type Label = string;
/**
 * must have same length as this.pcm | this.completedPcm
 */
type PcmLabels = readonly Label[];
type ComparisonRatio = PosReal | 0 | typeof NaN;
type Weights = readonly number[];

/**
 * ideally a valid `PairwiseComparisonMatrix`, but may be too small, non-square, disconnected (representing a disconnected graph), or contain illegal (eg negative numbers) values.
 */
type UnvalidatedPCM = ReadonlyDeep<number[][]>;

/**
 * a square, initallyValidated PCM but potentially with isolated nodes and disconnected
 */
type MaybeDisconnectedPCM = ReadonlyDeep<number[][]>;

/**
 * An explicitly possibly incomplete `PairwiseComparisonMatrix`, with `NaN`s where ratios have not been explicited defined or computed.
 * @remarks must be connected (represent a connected graph)
 */
type IncompletePCM = ReadonlyDeep<ComparisonRatio[][]>;

/**
 * A `PairwiseComparisonMatrix` with legal, positive real number ratio estimates for each pairwise relationship.
 */
type CompletePCM = ReadonlyDeep<PosReal[][]>;

/**
 * A numeric square matrix, abbreviated `PCM`, that stores relative estimates between items for some quantitative attribute.
 * Explicit comparisons are positive real numbers, and empty comparisons are denoted by `0`s or `NaN`s (preferred).
 * @example
 * If we are estimating the relative probability of outcomes A,B,& C, the resulting PCM might look like:
 * ```
 * PCM = [[1, 1.5, 2.2],
 *        [.65, 1, 1.5],
 *        [.4, .7, 1]]
 * ```
 * where PCM[0][1] = 1.5 means that A is 1.5 more (in this case, 'likely') than B.
 * @remarks the ratios in a PCM generally represent initial subjective estimates and need not be perfectly internally coherent.
 * @remarks a PCM must represent a connected graph.
 * @see https://en.wikipedia.org/wiki/Pairwise_comparison
 */
type PairwiseComparisonMatrix = IncompletePCM | CompletePCM;
type PCM = PairwiseComparisonMatrix;

class DisconnectedGraphError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "DisconnectedGraphError";
    }
}

/**
 * @returns errors if `ideallyPcm` is not length > 2 and square
 */
function initiallyValidatePcm(ideallyPcm: UnvalidatedPCM): Error[] {
    const errors = [];
    if (ideallyPcm.length < 2) errors.push(new Error(`pcm too short`));
    if (!ArrMatrix.isSquare(ideallyPcm)) {
        errors.push(new Error(`pcm not square!`));
    }
    // if (!matrix.isConnected(ideallyPcm)) errors.push(new Error(`pcm is not connected!`))
    return errors;
}

/**
 * @returns errors if pcm is not length \> 2, square, and if labels.length != pcm.length
 */
function initallyValidatePcmAndLabels(
    ideallyPcm: UnvalidatedPCM,
    labels: readonly Label[],
): Error[] {
    const errors = initiallyValidatePcm(ideallyPcm);
    if (ideallyPcm.length !== labels.length) {
        errors.push(
            new Error(`there must be a label for each item represented in pcm`),
        );
    }
    return errors;
}

function selfComparisonEq1(x: ComparisonRatio, i: Index, j: Index): 1 | number {
    return i === j ? 1 : x;
}
function toLegalComparisonRatio(x: number): PosReal | typeof NaN {
    return Number.isFinite(x) && x > 0 ? x : NaN;
}

function isCompletePcm(pcm: PairwiseComparisonMatrix): boolean {
    return pcm.every((row) => row.every((x) => Number.isFinite(x) && x > 0));
}

function makeLabeledPcm(
    pcm: PairwiseComparisonMatrix,
    pcmLabels: PcmLabels = Letters.arr12.slice(0, pcm.length),
): {
    [pcmLabel: string]: {
        [pcmLabel: string]: number;
    };
} {
    return Object.fromEntries(
        zip(
            pcmLabels,
            pcm.map((row) => Object.fromEntries(zip(pcmLabels, row))),
        ),
    );
}

/**
 * @returns copies of pcm & labels with isolated nodes removed. still may be disconnected.
 */
function withIsolatedNodesRemoved(
    maybeDisconnectedPcm: MaybeDisconnectedPCM,
    pcmLabels: PcmLabels,
): { pcm: MaybeDisconnectedPCM; pcmLabels: PcmLabels } {
    console.assert(
        maybeDisconnectedPcm.length === pcmLabels.length,
        "dev error, labels should already be off same length",
    );

    const pcm = ArrMatrix.clone(maybeDisconnectedPcm) as WritableDeep<
        IncompletePCM
    >;
    const labels = Array.from(pcmLabels);
    const degrees = ArrMatrix.adjacencyMatrix(pcm).map((row) => sum(row));
    const idxToRemove = degrees
        .map((x, i) => (x === 0 ? i : false))
        .filter((x) => x !== false)
        .reverse() as number[];
    for (const idx of idxToRemove) {
        // console.log(`index ${idx} removed from matrix`);
        for (const row of pcm) {
            row.splice(idx, 1);
        }
        pcm.splice(idx, 1); // remove row
        if (labels) labels.splice(idx, 1);
    }

    console.assert(
        pcm.length === labels.length,
        "dev error, this function is messed up",
    );
    if (idxToRemove.length > 0) {
        console.log(`${idxToRemove.length} items removed from matrix`);
    }
    return {
        pcm,
        pcmLabels: labels,
    };
}

/**
 * If both 'mirror' comparisons exist (A=Bx and B=Ax), complete them with their geometric mean and its reciprocal.
 * If only one side of a comparison exist, complete its mirror with its reciprocal.
 * @returns a new incomplete PCM with averaged, consistent reciprocal comparisons.
 */
function withReciprocalsAveraged(
    iPcm: IncompletePCM | MaybeDisconnectedPCM,
): IncompletePCM {
    const M = ArrMatrix.clone(iPcm) as WritableDeep<IncompletePCM>;
    for (let i = 0; i < M.length; i++) {
        for (let j = i + 1; j < M.length; j++) {
            const aBx = M[i][j]; // `A = Bx`
            const bAdx = M[j][i]; // `B = A/x`
            if (aBx && bAdx) {
                const geoAvg = Math.sqrt(aBx / bAdx);
                M[i][j] = geoAvg;
                M[j][i] = 1 / geoAvg;
            } else if (aBx) {
                M[j][i] = 1 / aBx;
            } else if (bAdx) {
                M[i][j] = 1 / bAdx;
            }
        }
    }
    return M;
}

/**
 * @param completePcm - A `PairwiseComparisonMatrix` with legal, positive real number ratio estimates for each pairwise relationship.
 * @returns weights of solved PCM, normalized to unity
 */
function completePcmSolver(completePcm: CompletePCM): Weights {
    const rowGeoMeans = completePcm.map((row) => geometricMean(row));
    return unityNormalize(rowGeoMeans);
}

/**
 * @returns weights of solved PCM, normalized to unity
 * @remarks variable names and logic based on method Incomplete Geometric Mean Method in https://doi.org/10.3390/math8111873
 */
function incompletePcmSolver(iPcm: IncompletePCM): Weights {
    const AInv = ArrMatrix.inverse(
        ArrMatrix.add(
            ArrMatrix.laplacian(iPcm),
            ArrMatrix.ones(iPcm.length, iPcm[0].length),
        ),
    );
    const r = iPcm
        .map((row) => row.reduce((acc, x) => (x > 0 ? acc + Math.log(x) : acc), 0))
        .map((x) => [x]);
    const w = ArrMatrix.multiply(AInv, r)
        .flat()
        .map((x) => Math.exp(x));
    return unityNormalize(w);
}

/**
 * @param initPcm - Pairwise comparison matrix to solve and complete
 * @param initPcmLabels - labels for items in `initPcm` (important to keep track of as isolated nodes will be removed)
 * @returns solved pcmWeights, completedPcm, pcmLabels, labeledWeights, and labeledPcm, all with any isolated nodes removed else a list of errors
 */
export function solve(
    initPcm: UnvalidatedPCM,
    initPcmLabels: readonly Label[] = Letters.arr12.slice(0, initPcm.length),
):
    | {
        pcmWeights: Weights;
        completedPcm: CompletePCM;
        pcmLabels: PcmLabels;
        labeledWeights: { [pcmLabel: string]: Weights[0] };
        labeledPcm: ReturnType<typeof makeLabeledPcm>;
    }
    | Error[] {
    const maybeErrors = initallyValidatePcmAndLabels(initPcm, initPcmLabels);
    if (maybeErrors.length > 0) return maybeErrors;

    initPcm = withReciprocalsAveraged(initPcm);

    let { pcm, pcmLabels } = withIsolatedNodesRemoved(initPcm, initPcmLabels);
    if (!ArrMatrix.isConnected(pcm as Square<MatrixT>)) {
        return maybeErrors.concat(
            new DisconnectedGraphError("pcm is disconnected"),
        );
    }

    pcm = ArrMatrix.with(pcm, selfComparisonEq1, toLegalComparisonRatio);
    pcm = withReciprocalsAveraged(pcm); // probably not necessary now since done earlier

    const pcmWeights = isCompletePcm(initPcm)
        ? completePcmSolver(initPcm)
        : incompletePcmSolver(initPcm);

    const completedPcm = pcm.map((row, i) =>
        row.map((x, j) => (Object.is(x, NaN) ? pcmWeights[i] / pcmWeights[j] : x))
    );

    const labeledWeights = Object.fromEntries(zipIters(pcmLabels, pcmWeights));

    const labeledPcm = makeLabeledPcm(pcm, pcmLabels);

    return {
        pcmWeights,
        completedPcm,
        pcmLabels,
        labeledWeights,
        labeledPcm,
    };
}

type QueryIdx = readonly [Index, Index];
type Query = readonly [Label, Label];

function nextOptimalQueryToComplete(
    pcm: MaybeDisconnectedPCM,
    pcmLabels: PcmLabels,
): Query {
    pcm = withReciprocalsAveraged(pcm);
    if (isCompletePcm(pcm)) throw new Error("pcm is already completed!");

    const subgraphs = ArrMatrix.getSubgraphs(pcm).sort((b, a) =>
        a.length - b.length
    );
    if (subgraphs.length > 1) {
        return [pcmLabels[subgraphs[0][0]], pcmLabels[subgraphs[1][0]]];
    }
    /* pcm is connected */

    const candidates: QueryIdx[] = [];
    for (const [i, row] of pcm.entries()) {
        for (const [j, x] of row.entries()) {
            if (Object.is(x, NaN)) candidates.push([i, j]);
        }
    }
    // TODO: read up on proper algorithms for this
}

function optimalQueriesToComplete(pcm: MaybeDisconnectedPCM): Query[] {
    pcm = withReciprocalsAveraged(pcm);
    // if pcm is disconnected, I think we want to connect the largest disconnected subgraphs
    const subgraphs = ArrMatrix.getSubgraphs(pcm);
}

/**
 * @param n - size of matrix to return
 * @param consistencyFactor - on [0,1]. higher means more consistency
 * @returns - new random pairwise comparison matrix
 */
function randPcm(n: number, consistencyFactor = 0.8): PairwiseComparisonMatrix {
    const randFactor = (i: number, j: number): number =>
        i === j ? 1 : randFloat(1, 9) **
            ((Math.random() < consistencyFactor ? 1 : -1) * Math.sign(i - j));

    return ArrMatrix.new(n, n, randFactor);
}

export const pcm = {
    solve,
    randPcm,
};

function tests() {
    const M = [
        [
            [1, 1.2, 0.5, 2.5, 0.62],
            [0.9, 1, 0.35, 2.5, 0.5],
            [2.2, 2.3, 1, 7, 1.22],
            [0.3, 0.4, 1 / 8, 1, 0.65],
            [1.3, 1.4, 0.85, 7, 1],
        ],

        [
            [1, 1.2, 0.5, 2.5, 0.62],
            [0.9, 1, 0.35, 2.5, 0.5],
            [2.2, 2.3, 1, 7, 1.22],
            [0.3, 0.4, 1 / 8, 1, 0],
            [1.3, 1.4, 0.85, 0, 1],
        ], // 4->5 & 5->4 are incomplete

        [
            [NaN, 1.2, NaN, 2.5, 0.62],
            [0.9, NaN, 0.35, 2.5, NaN],
            [NaN, 2.3, NaN, 7, 1.22],
            [0.3, 0.4, 1 / 8, NaN, 0.65],
            [1.3, NaN, 0.85, 7, NaN],
        ],

        [
            [1, 1.2, 0.5, 2.5, 0, 0.62],
            [0.9, 1, 0.35, 2.5, 0, 0.5],
            [2.2, 2.3, 1, 7, 0, 1.22],
            [0.3, 0.4, 1 / 8, 1, 0, 0.65],
            [0, 0, 0, 0, 1, 0],
            [1.3, 1.4, 0.85, 7, 0, 1],
        ],
    ];

    const cars = [
        [1, 0, 0, 2],
        [0, 1, 3, 0],
        [0, 1 / 3, 1, 2],
        [1 / 2, 0, 1 / 2, 1],
    ];
    const carsProperResult = [
        0.181_818_181_818_181_85,
        0.545_454_545_454_545_2,
        0.181_818_181_818_181_8,
        0.090_909_090_909_090_94,
    ];
    const carsCalcedResult = pcm.solve(cars);
    console.log(carsCalcedResult); // should be ~= [.18,.54,.18,.09]
    console.assert(
        dequal(carsProperResult, carsCalcedResult?.pcmWeights),
        "calced result deviating from expected",
    );
}
// tests()

