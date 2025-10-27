# OATest: Optimization-Aware Test Generation for DL Compilers



- ### Reproducibility

  #### 0. File Structure

  The project directory includes the following items:

  * TVM: source code for fuzzing TVM
  * ORT: source code for fuzzing ONNXRuntime
  * res: save the original dataset and experiment results
  * experiments: plot the results in the paper

  ####  1. Build Environment

  > Install Software under test by the official documentats
  * TVM (commit id: 292ecfd):   build from the source guided by the [documentation](https://tvm.apache.org/docs/install/from_source.html)
  * ONNXRuntime (commit id: 5c1b7cc):   build from the source following the [documentation](https://onnxruntime.ai/docs/build/eps.html#amd-rocm)

  > Install the dependency python package for OATest
  ```powershell
  pip install requirements.txt
  ```

  

  #### 2. Run the fuzzing

  ```
  python run_fuzz_all.py 
  ```
  ----

  

  ### 3. Bug Details

  OATest has been detected **56** previously unknown bugs, **42/24** of which have been confirmed/fixed by developers. Below, we present the specifics of each bug detected by OATest!

  

  | Bug ID | Compiler    | Symptom       | Root Cause                   | Optimization                           | Issue  | Status    |                        Sub Root Cause |
  | ------ | ----------- | ------------- | ---------------------------- | -------------------------------------- | ------ | --------- | ------------------------------------: |
  | 1      | TVM         | Crash         | Incorrect Code Logic         | MergeCompositeFunctions                | #17120 | Fixed     |     Incorrect Optimization Code Logic |
  | 2      | TVM         | Crash         | Tensor Shape Problem         | DeadCodeElimination                    | #17121 | Fixed     |                                       |
  | 3      | TVM         | Crash         | Incorrect Exception Handling | LegalizeOps                            | #17175 | Confirmed |                                       |
  | 4      | TVM         | Crash         | Incorrect Code Logic         | AttachGlobalSymbols                    | #17176 | Fixed     |     Incorrect Optimization Code Logic |
  | 5      | TVM         | Crash         | Incorrect Code Logic         | LiftTransformParams                    | #17200 | Fixed     |     Incorrect Optimization Code Logic |
  | 6      | TVM         | Inconsistency | Incorrect Code Logic         | RealizeVDevice, RealizeVDevice         | #17205 | Fixed     |     Incorrect Optimization Code Logic |
  | 7      | TVM         | Crash         | Incorrect Code Logic         | MergeCompositeFunctions                | #17210 | Fixed     |     Incorrect Optimization Code Logic |
  | 8      | TVM         | Crash         | Type Problem                 | -                                      | #17211 | Fixed     |                           Tensor Type |
  | 9      | TVM         | Crash         | Incorrect Code Logic         | RealizeVDevice, RealizeVDevice         | #17213 | Fixed     |     Incorrect Optimization Code Logic |
  | 10     | TVM         | Crash         | Tensor Shape Problem         | FlattenBuffer                          | #17215 | Fixed     |                                       |
  | 11     | TVM         | Crash         | Tensor Shape Problem         | VMBuiltinLower                         | #17217 | Fixed     |                                       |
  | 12     | TVM         | Crash         | Tensor Shape Problem         | VMBuiltinLower                         | #17218 | Fixed     |                                       |
  | 13     | TVM         | Crash         | Incorrect Code Logic         | DeadCodeElimination                    | #17222 | Fixed     | Incorrect Non-optimization Code Logic |
  | 14     | TVM         | Crash         | Incorrect Exception Handling | -                                      | #17223 | Confirmed |                                       |
  | 15     | TVM         | Crash         | Incorrect Code Logic         | LiftTransformParams                    | #17231 | Fixed     |     Incorrect Optimization Code Logic |
  | 16     | TVM         | Crash         | Type Problem                 | -                                      | #17235 | Fixed     |                conventional data type |
  | 17     | TVM         | Crash         | Incorrect Exception Handling | -                                      | #17242 | Fixed     |                                       |
  | 18     | TVM         | Crash         | Type Problem                 | -                                      | #17243 | Fixed     |                conventional data type |
  | 19     | TVM         | Inconsistency | Incorrect Code Logic         | -                                      | #17249 | Fixed     | Incorrect Non-optimization Code Logic |
  | 20     | TVM         | Inconsistency | Incorrect Code Logic         | RemoveUnusedOutputs                    | #17253 | Fixed     |     Incorrect Optimization Code Logic |
  | 21     | TVM         | Crash         | Type Problem                 | ToMixedPrecision                       | #17254 | Fixed     |                           Tensor Type |
  | 22     | TVM         | Crash         | Incorrect Code Logic         | LazyTransformParams                    | #17269 | Confirmed |     Incorrect Optimization Code Logic |
  | 23     | TVM         | Crash         | Incorrect Exception Handling | -                                      | #17311 | Confirmed |                                       |
  | 24     | TVM         | Crash         | Incorrect Code Logic         | KillAfterLastUse, FoldConstant         | #17340 | Confirmed |     Incorrect Optimization Code Logic |
  | 25     | TVM         | Crash         | Incorrect Code Logic         | AnnotateTIROpPattern,FuseOps,FuseTIR   | #17341 | Confirmed |     Incorrect Optimization Code Logic |
  | 26     | TVM         | Crash         | Incorrect Code Logic         | StaticPlanBlockMemory                  | #17348 | Confirmed |     Incorrect Optimization Code Logic |
  | 27     | TVM         | Crash         | Incorrect Code Logic         | FuseTIR, LambdaLift, AllocateWorkspace | #17357 | Confirmed |     Incorrect Optimization Code Logic |
  | 28     | TVM         | Crash         | Incorrect Exception Handling | LegalizeOps                            | #17370 | Fixed     |                                       |
  | 29     | TVM         | Crash         | -                            | LambdaLift                             | #17389 | Awaiting  |                                       |
  | 30     | TVM         | Crash         | -                            | -                                      | #17390 | Awaiting  |                                       |
  | 31     | TVM         | Crash         | Type Problem                 | LambdaLift                             | #17406 | Awaiting  |                                       |
  | 32     | TVM         | Crash         | -                            | LiftTransformParams                    | #17460 | Awaiting  |                                       |
  | 33     | TVM         | Crash         | -                            | -                                      | #17478 | Awaiting  |                                       |
  | 34     | TVM         | Crash         | -                            | InlinePrivateFunctions                 | #17479 | Awaiting  |                                       |
  | 35     | TVM         | Crash         | Incorrect Code Logic         | LegalizeOps                            | #17483 | Fixed     | Incorrect Non-optimization Code Logic |
  | 36     | TVM         | Crash         | Incorrect Code Logic         | LegalizeOps                            | #17486 | Fixed     | Incorrect Non-optimization Code Logic |
  | 37     | TVM         | Crash         | Incorrect Code Logic         | -                                      | #17487 | Confirmed | Incorrect Non-optimization Code Logic |
  | 38     | TVM         | Crash         | Incorrect Code Logic         | StaticPlanBlockMemory                  | #17488 | Fixed     |     Incorrect Optimization Code Logic |
  | 39     | TVM         | Crash         | -                            | LiftTransformParams                    | #17493 | Awaiting  |                                       |
  | 40     | TVM         | Crash         | -                            | LiftTransformParams                    | #17494 | Awaiting  |                                       |
  | 41     | ONNXRuntime | Crash         | Incorrect Exception Handling | FuseQuickGeLU                          | #23086 | Confirmed |                                       |
  | 42     | ONNXRuntime | Crash         | Incorrect Exception Handling | -                                      | #23088 | Awaiting  |                                       |
  | 43     | ONNXRuntime | Crash         | Incorrect Code Logic         | FusedConv                              | #23114 | Confirmed |     Incorrect Optimization Code Logic |
  | 44     | ONNXRuntime | Crash         | Type Problem                 | FuseReluClip                           | #23116 | Confirmed |                           Tensor Type |
  | 45     | ONNXRuntime | Crash         | Incorrect Code Logic         | DeadCodeElimination                    | #23118 | Confirmed |     Incorrect Optimization Code Logic |
  | 46     | ONNXRuntime | Crash         | Incorrect Exception Handling | FusedConv                              | #23119 | Confirmed |                                       |
  | 47     | ONNXRuntime | Inconsistency | -                            | -                                      | #23133 | Awaiting  |                                       |
  | 48     | ONNXRuntime | Crash         | Incorrect Code Logic         | GeluFusion                             | #23138 | Confirmed |     Incorrect Optimization Code Logic |
  | 49     | ONNXRuntime | Inconsistency | -                            | -                                      | #23142 | Awaiting  |                                       |
  | 50     | ONNXRuntime | Inconsistency | Incorrect Code Logic         | MemCpy                                 | #23143 | Confirmed |     Incorrect Optimization Code Logic |
  | 51     | ONNXRuntime | Inconsistency | Incorrect Code Logic         | Quantizer, Dequantizer                 | #23199 | Confirmed |     Incorrect Optimization Code Logic |
  | 52     | ONNXRuntime | Inconsistency | -                            | -                                      | #23209 | Awaiting  |                                       |
  | 53     | ONNXRuntime | Inconsistency | -                            | -                                      | #23212 | Awaiting  |                                       |
  | 54     | ONNXRuntime | Crash         | Incorrect Exception Handling | -                                      | #23213 | Awaiting  |                                       |
  | 55     | ONNXRuntime | Crash         | Incorrect Code Logic         | Dequantizer                            | #23258 | Fixed     |     Incorrect Optimization Code Logic |
  | 56     | ONNXRuntime | Inconsistency | Incorrect Code Logic         | -                                      | #23284 | Confirmed | Incorrect Non-optimization Code Logic |

  

