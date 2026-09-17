// ocl-probe - will ggml's OpenCL backend accept this device, and will it run?
//
//   cmake -B build -DNLLB_OPENCL=ON && cmake --build build --target ocl-probe
//   ./build/bin/ocl-probe
//
// WHY THIS EXISTS. Step 14 spent an afternoon on two OpenCL runtimes that both
// looked fine until they were not, and the two failures are at DIFFERENT
// stages:
//
//   pocl 5.0             registers?  NO  - no cl_khr_fp16, dropped at startup
//   Intel CPU runtime    registers?  YES - and then cannot COMPILE the kernels
//
// Neither is visible from `clinfo`, which reports extensions and stops. The
// second one is the expensive kind: ggml prints `device FP16 support: true`,
// you believe you have a working device, and the failure arrives later as a
// wall of JIT output. So this probe asks BOTH questions, per device, in the
// order ggml asks them, and prints a verdict rather than a capability dump.
//
// It links NOTHING - no ggml, no nllb, not even an ICD loader (it dlopen()s
// the vendor library at runtime) - which is the point:
// when it disagrees with ggml, the fault is in ggml or in this file, and when
// it agrees, the fault is in the runtime and there is nothing to fix here.
//
// Exit code is the number of devices that would WORK, so `ocl-probe; echo $?`
// is scriptable and 0 means "no usable OpenCL device on this machine".

#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

// --------------------------------------------------------------------------
// The loader is resolved at RUNTIME, and that is the point on Android.
// --------------------------------------------------------------------------
// Linking against an ICD loader would mean building one for every target
// before you can ask the question this tool exists to answer. On Android that
// is llama.cpp's whole OpenCL-ICD-Loader step - a cross-compiled dependency
// you set up BEFORE finding out whether the phone's GPU is usable at all,
// which is backwards.
//
// Every Android device with a Qualcomm GPU already ships the vendor's OpenCL
// library at /vendor/lib64/libOpenCL.so. So: dlopen it, resolve the eleven
// entry points used below, and the probe becomes one self-contained binary
// with NO link-time OpenCL dependency. Cross-compile it with the NDK, `adb
// push`, run. No ICD loader, no root, no APK.
//
// The headers are still needed at COMPILE time for the types. They are
// header-only (KhronosGroup/OpenCL-Headers) and the NDK does not ship them.
#if defined(_WIN32)
#  include <windows.h>
#  define OCL_DLOPEN(n)     ((void *) LoadLibraryA(n))
#  define OCL_DLSYM(h, s)   ((void *) GetProcAddress((HMODULE) (h), (s)))
static const char * const OCL_LIBS[] = { "OpenCL.dll", NULL };
#else
#  include <dlfcn.h>
#  define OCL_DLOPEN(n)     dlopen((n), RTLD_NOW | RTLD_LOCAL)
#  define OCL_DLSYM(h, s)   dlsym((h), (s))
static const char * const OCL_LIBS[] = {
    "libOpenCL.so",                  // the ICD loader, if one is installed
    "libOpenCL.so.1",
    "/vendor/lib64/libOpenCL.so",    // Android, arm64 - Qualcomm ships this
    "/system/vendor/lib64/libOpenCL.so",
    "/vendor/lib/libOpenCL.so",      // Android, 32-bit
    "/system/lib64/libOpenCL.so",
    "/System/Library/Frameworks/OpenCL.framework/OpenCL",  // macOS
    NULL
};
#endif

// Only what this file calls. Keep this list minimal: every symbol added here
// is one more thing that can be absent on a device and turn a diagnostic into
// a crash.
#define OCL_FUNCS(X)              \
    X(clGetPlatformIDs)           \
    X(clGetPlatformInfo)          \
    X(clGetDeviceIDs)             \
    X(clGetDeviceInfo)            \
    X(clCreateContext)            \
    X(clCreateProgramWithSource)  \
    X(clBuildProgram)             \
    X(clGetProgramBuildInfo)      \
    X(clCreateKernel)             \
    X(clReleaseKernel)            \
    X(clReleaseProgram)           \
    X(clReleaseContext)

#define X(name) static __typeof__(name) * p_##name = NULL;
OCL_FUNCS(X)
#undef X

static const char * g_lib = NULL;

// NOTE THE ORDER. This function must come BEFORE the shadowing #defines
// below, because `__typeof__(name)` here has to resolve against the REAL
// prototype from <CL/cl.h>. Once `clGetPlatformIDs` means `p_clGetPlatformIDs`,
// __typeof__ yields the POINTER type and the cast silently becomes
// pointer-to-pointer. gcc caught it as -Wincompatible-pointer-types; moving
// the block is the fix, not casting through void*, which would have compiled
// clean and been just as wrong.
static int ocl_load(void) {
    void * h = NULL;
    for (int i = 0; OCL_LIBS[i]; ++i) {
        h = OCL_DLOPEN(OCL_LIBS[i]);
        if (h) { g_lib = OCL_LIBS[i]; break; }
    }
    if (!h) return 0;

    // A library that opens but is missing an entry point is worse than one
    // that does not open, because the failure would be a null call later. Bail
    // as a unit.
#define X(name) \
    p_##name = (__typeof__(name) *) OCL_DLSYM(h, #name); \
    if (!p_##name) { fprintf(stderr, "%s: no symbol %s\n", g_lib, #name); return 0; }
    OCL_FUNCS(X)
#undef X
    return 1;
}

// Shadow the real names so everything below reads like ordinary OpenCL code.
#define clGetPlatformIDs          p_clGetPlatformIDs
#define clGetPlatformInfo         p_clGetPlatformInfo
#define clGetDeviceIDs            p_clGetDeviceIDs
#define clGetDeviceInfo           p_clGetDeviceInfo
#define clCreateContext           p_clCreateContext
#define clCreateProgramWithSource p_clCreateProgramWithSource
#define clBuildProgram            p_clBuildProgram
#define clGetProgramBuildInfo     p_clGetProgramBuildInfo
#define clCreateKernel            p_clCreateKernel
#define clReleaseKernel           p_clReleaseKernel
#define clReleaseProgram          p_clReleaseProgram
#define clReleaseContext          p_clReleaseContext

// The options ggml-opencl builds EVERY kernel with. They are not neutral:
// step 14 measured `exp(half)` compiling bare and failing to link under these
// on Intel's CPU runtime, which is the whole reason this probe compiles
// something instead of only reading extension strings.
#define GGML_CL_OPTS \
    "-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations " \
    "-cl-finite-math-only -cl-fast-relaxed-math"

// Small and representative: a half math builtin, which is the exact shape of
// what ggml's glu.cl asks for and the exact thing that was missing.
static const char * KSRC =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void probe(__global half * o) {\n"
    "    o[0] = exp(o[0]) + (half) 1.0h;\n"
    "}\n";

static char * info_str(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (clGetDeviceInfo(d, what, 0, NULL, &n) != CL_SUCCESS || n == 0) {
        char * empty = (char *) malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    char * s = (char *) malloc(n + 1);
    if (!s) return NULL;
    if (clGetDeviceInfo(d, what, n, s, NULL) != CL_SUCCESS) { s[0] = '\0'; return s; }
    s[n] = '\0';
    return s;
}

static const char * type_name(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU)         return "GPU";
    if (t & CL_DEVICE_TYPE_CPU)         return "CPU";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    return "other";
}

int main(void) {
    if (!ocl_load()) {
        printf("no OpenCL library could be loaded.\n\n"
               "Tried: ");
        for (int i = 0; OCL_LIBS[i]; ++i) printf("%s%s", i ? ", " : "", OCL_LIBS[i]);
        printf("\n\n"
               "  Android: the vendor library is /vendor/lib64/libOpenCL.so. If it is\n"
               "           absent, this SoC has no OpenCL driver - Mali and Xclipse\n"
               "           devices often ship Vulkan only.\n"
               "  Linux:   install an ICD loader (ocl-icd-libopencl1).\n"
               "  Windows: the vendor driver installs OpenCL.dll.\n");
        return 0;
    }
    printf("loader: %s\n", g_lib);

    cl_uint n_plat = 0;
    if (clGetPlatformIDs(0, NULL, &n_plat) != CL_SUCCESS || n_plat == 0) {
        printf("no OpenCL platform at all.\n\n"
               "That is the ICD LOADER finding no driver, not a missing GPU.\n"
               "  Windows: the vendor driver installs the ICD; check the registry key\n"
               "           HKLM\\SOFTWARE\\Khronos\\OpenCL\\Vendors\n"
               "  Linux:   /etc/OpenCL/vendors/*.icd\n");
        return 0;
    }

    cl_platform_id plats[16];
    if (n_plat > 16) n_plat = 16;
    clGetPlatformIDs(n_plat, plats, NULL);

    printf("%u OpenCL platform(s)\n", n_plat);
    printf("ggml compiles kernels with: %s\n\n", GGML_CL_OPTS);

    int usable = 0, seen = 0;

    for (cl_uint p = 0; p < n_plat; ++p) {
        char pname[256] = "?";
        clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);

        cl_uint n_dev = 0;
        // CL_DEVICE_TYPE_ALL, because that is what ggml enumerates - a
        // CPU-backed OpenCL device is a device to it, and it will present as
        // a GPU regardless of what the driver says.
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &n_dev) != CL_SUCCESS || n_dev == 0) {
            printf("platform %u: %s\n    no devices\n\n", p, pname);
            continue;
        }
        cl_device_id devs[16];
        if (n_dev > 16) n_dev = 16;
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, n_dev, devs, NULL);

        printf("platform %u: %s\n", p, pname);

        for (cl_uint i = 0; i < n_dev; ++i) {
            seen++;
            cl_device_id d = devs[i];

            char * dname = info_str(d, CL_DEVICE_NAME);
            char * dver  = info_str(d, CL_DEVICE_VERSION);
            char * ext   = info_str(d, CL_DEVICE_EXTENSIONS);
            cl_device_type dt = 0;
            clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(dt), &dt, NULL);

            printf("  device %u: %s\n", i, dname ? dname : "?");
            printf("    %s   %s\n", type_name(dt), dver ? dver : "?");

            // --- gate 1: what ggml checks before it will register the device.
            const int fp16 = ext && strstr(ext, "cl_khr_fp16") != NULL;
            const int sub   = ext && (strstr(ext, "cl_khr_subgroups")   != NULL ||
                                      strstr(ext, "cl_intel_subgroups") != NULL);
            // Subgroups are mandatory in OpenCL 2.x and optional in 3.0, so
            // ggml only demands the extension on a 3.0 device.
            const int is30  = dver && strstr(dver, "OpenCL 3.") != NULL;
            const int need_sub = is30;

            printf("    cl_khr_fp16                        %s\n", fp16 ? "yes" : "NO");
            printf("    cl_khr_subgroups/cl_intel_subgroups %s%s\n",
                   sub ? "yes" : "NO", need_sub ? "" : "  (not required below OpenCL 3.0)");

            if (!fp16 || (need_sub && !sub)) {
                printf("    VERDICT: ggml DROPS this device at startup.\n");
                printf("             You will see `ggml_opencl: device does not support FP16`\n"
                       "             (or `does not support subgroups`) on stderr, and then\n"
                       "             `--backend opencl` failing with the DEVICE LIST - a\n"
                       "             message about naming, for a cause about extensions.\n");
                if (dname && (strstr(dname, "NVIDIA") || strstr(dname, "GeForce") ||
                              strstr(dname, "RTX")    || strstr(dname, "GTX"))) {
                    printf("             EXPECTED on NVIDIA: their OpenCL does not claim\n"
                           "             conformance for cl_khr_fp16 and has no half math\n"
                           "             builtins. Use the CUDA backend for this GPU.\n");
                }
                printf("\n");
                free(dname); free(dver); free(ext);
                continue;
            }

            // --- gate 2: the one ggml does not check and that decides whether
            // anything runs. Registration is necessary and not sufficient.
            cl_int err = CL_SUCCESS;
            cl_context ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &err);
            if (!ctx || err != CL_SUCCESS) {
                printf("    VERDICT: registers, but clCreateContext failed (%d).\n\n", err);
                free(dname); free(dver); free(ext);
                continue;
            }

            cl_program prog = clCreateProgramWithSource(ctx, 1, &KSRC, NULL, &err);
            err = clBuildProgram(prog, 1, &d, GGML_CL_OPTS, NULL, NULL);

            if (err != CL_SUCCESS) {
                printf("    half math builtin under ggml's flags   NO (clBuildProgram %d)\n", err);
                size_t ln = 0;
                clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, 0, NULL, &ln);
                char * log = (char *) malloc(ln + 1);
                if (log) {
                    clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, ln, log, NULL);
                    log[ln] = '\0';
                    printf("    --- build log ---\n");
                    // A few hundred characters is enough to name the symbol.
                    for (size_t k = 0; k < ln && k < 600; ++k) putchar(log[k]);
                    printf("\n    -----------------\n");
                    free(log);
                }
                printf("    VERDICT: ggml REGISTERS this device and then fails loading\n"
                       "             kernels. It compiles its ENTIRE kernel set at backend\n"
                       "             init, so a gap in a kernel your model never reaches\n"
                       "             still takes the whole backend down. Not usable.\n\n");
            } else {
                // Creating the kernel is a separate step from building the
                // program, and Intel's CPU runtime failed at THAT one - a
                // green clBuildProgram is not the end of the question.
                cl_kernel k = clCreateKernel(prog, "probe", &err);
                if (err != CL_SUCCESS) {
                    printf("    half math builtin under ggml's flags   NO (clCreateKernel %d)\n", err);
                    printf("    VERDICT: program builds, kernel does not materialise.\n"
                           "             Same outcome as a build failure. Not usable.\n\n");
                } else {
                    printf("    half math builtin under ggml's flags   yes\n");
                    printf("    VERDICT: USABLE. ggml should register this as GPUOpenCL.\n\n");
                    usable++;
                    clReleaseKernel(k);
                }
            }
            clReleaseProgram(prog);
            clReleaseContext(ctx);
            free(dname); free(dver); free(ext);
        }
    }

    printf("%d of %d device(s) usable by ggml's OpenCL backend.\n", usable, seen);
    if (usable == 0) {
        printf("\nNo usable OpenCL device. This is a statement about the RUNTIMES\n"
               "installed here, not about nllb.cpp - build with --backend cpu, or\n"
               "use a backend whose driver is complete (CUDA on NVIDIA, Metal on\n"
               "Apple). See BACKEND.md, step 14.\n");
    } else if (usable > 1) {
        printf("\nNOTE: ggml enumerates devices on ONE PLATFORM ONLY. With more than\n"
               "one usable platform the others are invisible rather than merely\n"
               "unselected - set GGML_OPENCL_PLATFORM to choose.\n");
    }
    return usable;
}
