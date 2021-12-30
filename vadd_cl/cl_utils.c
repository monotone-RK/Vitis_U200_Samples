#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "cl_utils.h"

cl_platform_id platform = NULL;
cl_context ctx = NULL;
cl_program prg = NULL;
cl_command_queue cq = NULL;
cl_kernel kernel = NULL;
cl_int ret = 0;
cl_mem d_a = NULL;
cl_mem d_b = NULL;
cl_mem d_c = NULL;
cl_uint numdata = 0;

void init_ocl(const char *filename) {
  ///// Create platform //////
  cl_uint ret_num_platforms;
  ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
  fprintf(stderr, "Platforms Found: %d\n", ret_num_platforms);
  cl_platform_id *ocl_platforms = (cl_platform_id *)malloc(ret_num_platforms * sizeof(cl_platform_id));
  ret = clGetPlatformIDs(ret_num_platforms, ocl_platforms, NULL);
  // Select OpenCL platform for FPGA
  for (cl_uint i = 0; i < ret_num_platforms; i++) {
    char char_buffer[1024];
    clGetPlatformInfo(ocl_platforms[i], CL_PLATFORM_VERSION, 1024, char_buffer, NULL);
    fprintf(stderr, "Platform %d : %s\n", i, char_buffer);
    if (strcmp(char_buffer, "OpenCL 1.0") == 0) {
      platform = ocl_platforms[i];
    }
  }

  ///// Create context //////
  cl_device_id dev = NULL;
  cl_uint num_devs;
  ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &dev, &num_devs);
  assert(CL_SUCCESS == ret);
  ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &ret);
  assert(CL_SUCCESS == ret);

  ///// Create program /////
  // const char *filename = "./vadd.xclbin";
  FILE *fp = fopen(filename, "rb");
  assert((fp != NULL));
  fseek(fp, 0, SEEK_END);
  size_t binary_size = ftell(fp);
  unsigned char *binary = (unsigned char*)malloc(binary_size);
  rewind(fp);
  if (fread((void*)binary, binary_size, 1, fp) == 0) {
    free(binary);
    fclose(fp);
  }
  cl_int binary_status[num_devs];
  prg = clCreateProgramWithBinary(ctx, num_devs, &dev, (const size_t *)&binary_size,
                                  (const unsigned char **)&binary, binary_status, &ret);
  assert(CL_SUCCESS == ret);

  ///// Create buffer /////
  size_t const BUF_SIZE = sizeof(cl_int) * numdata;
  d_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, BUF_SIZE, NULL, &ret);
  assert(CL_SUCCESS == ret);
  d_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, BUF_SIZE, NULL, &ret);
  assert(CL_SUCCESS == ret);
  d_c = clCreateBuffer(ctx, CL_MEM_READ_WRITE, BUF_SIZE, NULL, &ret);
  assert(CL_SUCCESS == ret);

  ///// Create kernel /////
  kernel = clCreateKernel(prg, "vadd", &ret);
  assert(CL_SUCCESS == ret);

  ///// Set Kernel Args /////
  unsigned int argi = 0;
  ret = clSetKernelArg(kernel, argi++, sizeof(cl_mem), (void*)&d_a);
  assert(CL_SUCCESS == ret);
  ret = clSetKernelArg(kernel, argi++, sizeof(cl_mem), (void*)&d_b);
  assert(CL_SUCCESS == ret);
  ret = clSetKernelArg(kernel, argi++, sizeof(cl_mem), (void*)&d_c);
  assert(CL_SUCCESS == ret);
  ret = clSetKernelArg(kernel, argi++, sizeof(cl_uint), (void*)&numdata);
  assert(CL_SUCCESS == ret);

  ///// Create command queue /////
  cq = clCreateCommandQueue(ctx, dev, 0, &ret);
  assert(CL_SUCCESS == ret);
}

void cleanup_ocl() {
  // Delete command queue
  clReleaseCommandQueue(cq);

  // Delete kernel
  clReleaseKernel(kernel);

  // Delete buffer
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);

  // Delete program
  clReleaseProgram(prg);

  // Delete context
  clReleaseContext(ctx);
}
