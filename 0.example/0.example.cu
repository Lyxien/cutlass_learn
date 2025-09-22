#include <iostream>                                           // 标准输入输出流
#include "cutlass/gemm/device/gemm.h"                         // 引入cutlass头文件
 
using ColumnMajor = cutlass::layout::ColumnMajor;             // 列主序存储方式
using RowMajor    = cutlass::layout::RowMajor;                // 行主序存储方式
 
using CutlassGemm = cutlass::gemm::device::Gemm<float,        // A矩阵数据类型
                                                RowMajor,     // A矩阵存储方式
                                                float,        // B矩阵数据类型
                                                RowMajor,     // B矩阵存储方式
                                                float,        // C矩阵数据类型
                                                RowMajor>;    // C矩阵存储方式
 
void generate_tensor_2D(float *ptr, int i_M, int i_N){        // 二维矩阵填充函数（此处全部填充1）
    for(int i = 0; i < i_M; i++){
        for(int j = 0; j < i_N; j++){
            *(ptr + i*i_N + j ) = 1.0;
        }
    }
}
 
int main(int argc, const char *arg[]) {
 
    int M = 3840;           //M
    int N = 4096;           //N
    int K = 4096;           //K
 
    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;
 
    float alpha = 2.0;      //alpha
    float beta = 3.0;       //beta
 
    float *A;               //申明A矩阵host端指针
    float *B;               //申明B矩阵host端指针
    float *C;               //申明C矩阵host端指针
    float *D;               //申明D矩阵host端指针
 
    size_t A_mem_size = sizeof(float) * M * K; //memory size of matrix A = M * K * sizeof(float)
    size_t B_mem_size = sizeof(float) * K * N; //memory size of matrix B = K * N * sizeof(float)
    size_t C_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
    size_t D_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
 
    A = (float*)malloc(A_mem_size);  // host端A矩阵分配内存
    B = (float*)malloc(B_mem_size);  // host端B矩阵分配内存
    C = (float*)malloc(C_mem_size);  // host端C矩阵分配内存
    D = (float*)malloc(D_mem_size);  // host端D矩阵分配内存
 
    generate_tensor_2D(A, M, K);     // 填充A矩阵
    generate_tensor_2D(B, K, N);     // 填充B矩阵  
    generate_tensor_2D(C, M, N);     // 填充C矩阵  
 
    float *d_A;            // 申明device端A矩阵的指针
    float *d_B;            // 申明device端B矩阵的指针
    float *d_C;            // 申明device端C矩阵的指针
    float *d_D;            // 申明device端D矩阵的指针
 
    cudaMalloc((void**)&d_A, A_mem_size);  // device端为A矩阵分配内存
    cudaMalloc((void**)&d_B, B_mem_size);  // device端为B矩阵分配内存
    cudaMalloc((void**)&d_C, C_mem_size);  // device端为C矩阵分配内存
    cudaMalloc((void**)&d_D, D_mem_size);  // device端为D矩阵分配内存
 
    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); // 将矩阵A的数据传递到device端
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice); // 将矩阵B的数据传递到device端
    cudaMemcpy(d_C, C, C_mem_size, cudaMemcpyHostToDevice); // 将矩阵C的数据传递到device端
 
    CutlassGemm gemm_operator;                  // 申明cutlassgemm类
    CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {d_A, lda},     // source matrix A
                                {d_B, ldb},     // source matrix B
                                {d_C, ldc},     // source matrix C
                                {d_D, ldd},     // destination matrix D
                                {alpha, beta}); // alpha & beta
    gemm_operator(args); //运行Gemm
 
    cudaMemcpy(D, d_D, D_mem_size, cudaMemcpyDeviceToHost);  //将运行结果D矩阵传回host端
    std::cout << D[0] << std::endl;                          //打印D中第一行第一个数据
    std::cout << D[M * N - 1] << std::endl;                  //打印D中最后一行最后一个数据
 
    return 0;
}   
 