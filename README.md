# nvjpeg-bug-report-2

Demonstrates `nvjpegEncodeYUV` and `nvjpegEncodeImage` failing with status code `NVJPEG_STATUS_INTERNAL_ERROR` when encoding small images. The compiled executable compares all sizes 1 up to 32 in both dimensions, all output formats and subsampling factors, and with and without optimizing the Huffman table. These factors all play a role in whether the problem occurs or not. File `out.txt` shows an example run.

Tested with CUDA 12.3.2 driver version 545.23.08 on Ubuntu 22.04.4.

Not to be confused with https://github.com/nolmoonen/nvjpeg-bug-report.
