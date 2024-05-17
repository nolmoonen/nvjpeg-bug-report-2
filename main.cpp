#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define CUDA_CALL(call)                                                        \
  do {                                                                         \
    cudaError_t res_ = call;                                                   \
    if (cudaSuccess != res_) {                                                 \
      std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << "code=" << static_cast<unsigned int>(res_) << "("           \
                << cudaGetErrorString(res_) << ") \"" << #call << "\"\n";      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define NVJPEG_CALL(call)                                                      \
  do {                                                                         \
    nvjpegStatus_t res_ = call;                                                \
    if (NVJPEG_STATUS_SUCCESS != res_) {                                       \
      std::cout << "nvJPEG error at " << __FILE__ << ":" << __LINE__           \
                << " code=" << static_cast<unsigned int>(res_) << "\n";        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int dev_malloc(void *, void **ptr, size_t size, cudaStream_t) {
  return (int)cudaMalloc(ptr, size);
}

int dev_free(void *, void *ptr, size_t, cudaStream_t) {
  return (int)cudaFree(ptr);
}

int host_malloc(void *, void **ptr, size_t size, cudaStream_t) {
  return (int)cudaMallocHost(ptr, size);
}

int host_free(void *, void *ptr, size_t, cudaStream_t) {
  return (int)cudaFreeHost(ptr);
}

const char *css_to_cstr(nvjpegChromaSubsampling_t css) {
  switch (css) {
  case NVJPEG_CSS_444:
    return "4:4:4";
  case NVJPEG_CSS_422:
    return "4:2:2";
  case NVJPEG_CSS_420:
    return "4:2:0";
  case NVJPEG_CSS_440:
    return "4:4:0";
  case NVJPEG_CSS_411:
    return "4:1:1";
  case NVJPEG_CSS_410:
    return "4:1:0";
  default:
    return "unknown";
  }
}

const char *fmt_to_cstr(nvjpegOutputFormat_t fmt) {
  switch (fmt) {
  case NVJPEG_OUTPUT_YUV:
    return "YUV";
  case NVJPEG_OUTPUT_RGB:
    return "RGB";
  case NVJPEG_OUTPUT_BGR:
    return "BGR";
  case NVJPEG_OUTPUT_RGBI:
    return "RGBI";
  case NVJPEG_OUTPUT_BGRI:
    return "BGRI";
  default:
    return "unknown";
  }
}

void attempt_encode(int size_x, int size_y, bool opt_huffman,
                    nvjpegChromaSubsampling_t css, nvjpegOutputFormat_t fmt) {
  std::cout << "size:" << size_x << "x" << size_y
            << ",opt_huffman:" << std::boolalpha << opt_huffman;
  if (fmt == NVJPEG_OUTPUT_YUV) {
    std::cout << ",fmt:" << fmt_to_cstr(fmt) << ",css:" << css_to_cstr(css)
              << ": ";
  } else {
    std::cout << ",fmt:" << fmt_to_cstr(fmt) << ": ";
  }

  cudaStream_t stream = nullptr;

  nvjpegDevAllocatorV2_t dev_allocator = {&dev_malloc, &dev_free, nullptr};
  nvjpegPinnedAllocatorV2_t pinned_allocator = {&host_malloc, &host_free,
                                                nullptr};
  const int flags = 0;
  nvjpegHandle_t nvjpeg_handle;
  NVJPEG_CALL(nvjpegCreateExV2(NVJPEG_BACKEND_GPU_HYBRID, &dev_allocator,
                               &pinned_allocator, flags, &nvjpeg_handle));

  int num_channels = 1;
  int channel_mult = 1;
  switch (fmt) {
  case NVJPEG_OUTPUT_YUV:
    num_channels = 3;
    channel_mult = 1;
    break;
  case NVJPEG_OUTPUT_RGB:
    num_channels = 3;
    channel_mult = 1;
    break;
  case NVJPEG_OUTPUT_BGR:
    num_channels = 3;
    channel_mult = 1;
    break;
  case NVJPEG_OUTPUT_RGBI:
    num_channels = 1;
    channel_mult = 3;
    break;
  case NVJPEG_OUTPUT_BGRI:
    num_channels = 1;
    channel_mult = 3;
    break;
  default:
    std::cout << "unsupported format\n";
    std::exit(EXIT_FAILURE);
  }
  int mult_x = 1;
  int mult_y = 1;
  if (fmt == NVJPEG_OUTPUT_YUV) {
    switch (css) {
    case NVJPEG_CSS_444:
      mult_x = 1;
      mult_y = 1;
      break;
    case NVJPEG_CSS_422:
      mult_x = 2;
      mult_y = 1;
      break;
    case NVJPEG_CSS_420:
      mult_x = 2;
      mult_y = 2;
      break;
    case NVJPEG_CSS_440:
      mult_x = 1;
      mult_y = 2;
      break;
    case NVJPEG_CSS_411:
      mult_x = 4;
      mult_y = 1;
      break;
    case NVJPEG_CSS_410:
      mult_x = 4;
      mult_y = 2;
      break;
    default:
      std::cout << "unsupported subsampling\n";
      std::exit(EXIT_FAILURE);
    }
  }

  nvjpegImage_t img;
  for (int i = 0; i < num_channels; ++i) {
    // round up to multiple of subsampling factor for subsampled components
    int aw = i == 0 ? size_x : (size_x + mult_x - 1) / mult_x * mult_x;
    // round up to multiple of four, see
    //   https://forums.developer.nvidia.com/t/encode-to-chroma-subsampled-jpeg-fails-with-rgb-data/271684
    //   and https://developer.nvidia.com/bugs/4363416
    aw = (aw + 3) / 4 * 4;
    // round up to multiple of subsampling factor for subsampled components
    int ah = i == 0 ? size_y : (size_y + mult_y - 1) / mult_y * mult_y;
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&img.channel[i]), aw * ah));
    img.pitch[i] = aw;
  }

  nvjpegEncoderState_t encoder_state;
  NVJPEG_CALL(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, stream));
  nvjpegEncoderParams_t encoder_params;
  NVJPEG_CALL(
      nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, stream));

  NVJPEG_CALL(
      nvjpegEncoderParamsSetSamplingFactors(encoder_params, css, stream));

  // whether the huffman table is optimized or not, plays a role
  NVJPEG_CALL(nvjpegEncoderParamsSetOptimizedHuffman(
      encoder_params, opt_huffman ? 1 : 0, stream));

  NVJPEG_CALL(nvjpegEncoderParamsSetEncoding(
      encoder_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
  NVJPEG_CALL(nvjpegEncoderParamsSetQuality(encoder_params, 90, stream));

  if (fmt == NVJPEG_OUTPUT_YUV) {
    NVJPEG_CALL(nvjpegEncodeYUV(nvjpeg_handle, encoder_state, encoder_params,
                                &img, css, size_x, size_y, stream));
  } else {
    NVJPEG_CALL(nvjpegEncodeImage(
        nvjpeg_handle, encoder_state, encoder_params, &img,
        // cast is valid for RGB, BGR, RGBI, and BGRI
        static_cast<nvjpegInputFormat_t>(fmt), size_x, size_y, stream));
  }

  std::vector<unsigned char> buffer_out;
  size_t size_out = 0;
  NVJPEG_CALL(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL,
                                            &size_out, stream));
  buffer_out.resize(size_out);
  NVJPEG_CALL(nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle, encoder_state, buffer_out.data(), &size_out, stream));
  CUDA_CALL(cudaDeviceSynchronize());

  for (int i = 0; i < num_channels; ++i) {
    CUDA_CALL(cudaFree(img.channel[i]));
  }

  NVJPEG_CALL(nvjpegEncoderParamsDestroy(encoder_params));
  NVJPEG_CALL(nvjpegEncoderStateDestroy(encoder_state));

  NVJPEG_CALL(nvjpegDestroy(nvjpeg_handle));
  std::cout << "success\n";
}

void fork_attempt(int size_x, int size_y, bool opt_huffman,
                  nvjpegChromaSubsampling_t css, nvjpegOutputFormat_t fmt) {
  pid_t c_pid = fork();

  if (c_pid == -1) {
    perror("fork");
    std::exit(EXIT_FAILURE);
  }

  if (c_pid == 0) {
    attempt_encode(size_x, size_y, opt_huffman, css, fmt);
    std::exit(EXIT_SUCCESS);
  }

  wait(NULL); // await child
}

int main(int argc, char *argv[]) {
  int size_min = 1;
  int size_max = 32;

  for (int x = size_min; x <= size_max; ++x) {
    for (int y = size_min; y <= size_max; ++y) {
      for (bool opt_huffman : {false, true}) {
        for (const nvjpegChromaSubsampling_t css :
             {NVJPEG_CSS_444, NVJPEG_CSS_422, NVJPEG_CSS_420, NVJPEG_CSS_440,
              NVJPEG_CSS_411, NVJPEG_CSS_410}) {
          fork_attempt(x, y, opt_huffman, css, NVJPEG_OUTPUT_YUV);
        }
        for (const nvjpegOutputFormat_t fmt :
             {NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_BGR, NVJPEG_OUTPUT_RGBI,
              NVJPEG_OUTPUT_BGRI}) {
          fork_attempt(x, y, opt_huffman, NVJPEG_CSS_444, fmt);
        }
      }
    }
  }
}
