/*
 *  Created by meian on 2026/4/8.
 *
 *  Lamina Runtime Interface
 *
 *  不建议更改此文件，可以在每个发行包的include目录下找到此文件
 */
#pragma once
/*
 * 使用MSVC编译是不受支持的
 */
#if defined(_MSC_VER)
#error "MSVC is not supported, use clang-cl or other compilers in Windows"
#endif
#ifdef LMX_DLL
    #if defined(_WIN32) || defined(_WIN64)
        #ifdef LMX_BUILD
            #define LM_API __declspec(dllexport)
        #else
            #define LM_API __declspec(dllimport)
        #endif
    #else
        #define LM_API __attribute__((visibility("default")))
    #endif
#else
    #define LM_API
#endif

#define LMX_MAGIC_NUM   ((uint32_t)0x434D4C00)
#define LMX_VERSION     ((uint32_t)0x00000001)

#define LMX_INLINE __attribute__((always_inline)) inline
#if __cplusplus

extern "C" {
#endif
#include <stdint.h>
#include <stdio.h>

#define LM_CALL

/*
 * Lamina 接口状态管理
 */
struct LmLinkedNode;
typedef struct LmLinkedNode LmLinkedNode;
struct LmLinkedNode {
    void* ptr;
    LmLinkedNode* last;
};

struct LaminaVM;
typedef struct LaminaVM LaminaVM;
struct LmState {
    LmLinkedNode* n;
    LaminaVM* vm;
};
typedef struct LmState LmState;

extern LmState global_state;

LM_API LmState* lmx_newState();
LM_API void lmx_deleteState(const LmState* state);
/******************************/


struct LmValue;
typedef struct LmValue LmValue;

typedef int64_t LmInt;

struct LmModule;
typedef struct LmModule LmModule;




/*
 * lmx_doString
 *
 * Args:
 *     state (LmState*): 接口状态
 *     code  (const char*): 代码字符串指针
 *     name  (const char*): 模块名字
 * Return:
 *     LmModule*:  成功时返回加载完成的模块，失败nullptr
 *
 * Notes:
 *     失败原因可供参考： 编译错误，内存分配失败
 */
LM_API LmModule* LM_CALL lmx_doString(LmState* state, const char* code, const char* name);


/*
 * lmx_doFile
 *
 * Args:
 *     state (LmState*): 接口状态
 *     name  (const char*): 代码文件名字(同模块名)
 *     is_main_module (bool):  是不是主模块，主模块就是被直接运行的模块，不会被import
 * Return:
 *     LmModule*:  成功时返回加载完成的模块，失败nullptr
 *
 * Notes:
 *     失败原因可供参考： 编译错误，内存分配失败，已经存在主模块但是 is_main_module = true
 */
LM_API LmModule* LM_CALL lmx_doFile(LmState* state, const char* name, bool is_main_module);

LM_API void LM_CALL lmx_printASTFromFile(LmState* state, FILE* file, const char* name);

LM_API void LM_CALL lmx_printASTFromString(LmState* state, FILE* file, const char* code, const char* name);

LM_API void LM_CALL lmx_printMIRFromString(LmState* state, FILE* file, const char* code, const char* name);
LM_API void LM_CALL lmx_printMIRFromFile(LmState* state, FILE* file, const char* name);


LM_API bool LM_CALL lmx_moduleToFile(LmState* state, LmModule* module, const char* name);

LM_API LaminaVM* LM_CALL lmx_newLaminaVM(LmState* state, int argc, char** argv);

LM_API int LM_CALL lmx_vmRunModule(LmState* state, LaminaVM* vm, LmModule* module);

LM_API void LM_CALL lmx_vmEval(LmState* state, LaminaVM* vm, LmValue* result, const char* code);

#if __cplusplus
}
#endif
