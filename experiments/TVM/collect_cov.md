
## Collect coverage about TVM

#### 1.Capture coverage and save into xx.info
```shell script
lcov --capture --directory ./build/CMakeFiles/tvm_objs.dir/src/relax/transform --output-file cov_optfuzz.info --rc lcov_branch_coverage=1
lcov --capture --directory ./build/CMakeFiles/tvm_objs.dir/src/ --output-file cov_optfuzz.info --rc lcov_branch_coverage=1

```


#### 2. Delete history coverage
```shell script
find . -type f -name "*.gcda" -delete
```


#### 3. Generate html coverage report 
```shell script
genhtml cov_res/cov_optfuzz.info --output-directory cov_report --rc lcov_branch_coverage=1
```

---
---


## Collect coverage about ONNXRuntime

Optimization Dir: 

```
/software/onnxruntime/build/Linux/Release/CMakeFiles/onnxruntime_providers.dir/software/onnxruntime/onnxruntime/core
```


#### 1.Capture coverage and save into xx.info
```shell script
lcov --capture --directory  --output-file cov_mt_ort.info --rc lcov_branch_coverage=1

```
