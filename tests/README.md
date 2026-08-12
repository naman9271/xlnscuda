# xlnscuda tests

These tests are additive CUDA regression tests. Do not modify `../chkxlns16d.cu`;
that file remains the existing golden primitive regression test.

Current repository shape:

- `xlns16d.cu` and `xlns32d.cu` rely on typedefs/macros from the matching CPU
  implementation file.
- CPU-vs-GPU parity tests therefore include the matching CPU implementation
  from the `xlnscpp` submodule. Initialize it first with
  `git submodule update --init`.

Example from the `xlnscuda` repository root:

```sh
nvcc -std=c++11 tests/chkxlns16d_utils.cu -o /tmp/chkxlns16d_utils
nvcc -std=c++11 tests/chkxlns32d_utils.cu -o /tmp/chkxlns32d_utils
nvcc -std=c++11 tests/chkxlns16d_batch_convert.cu -o /tmp/chkxlns16d_batch_convert
nvcc -std=c++11 tests/chkxlns32d_batch_convert.cu -o /tmp/chkxlns32d_batch_convert

/tmp/chkxlns16d_utils
/tmp/chkxlns32d_utils
/tmp/chkxlns16d_batch_convert
/tmp/chkxlns32d_batch_convert
```
