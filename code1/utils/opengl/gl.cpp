#include "util/gl.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>


static void GLAPIENTRY GlDebugMessageCallback(
    GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam
)
{
    (void)id;
    (void)length;
    (void)userParam;
    // #define STR__(category, enum, var) var == GL_DEBUG ## _ ## category ## _ ## enum ? # de :
    // #define STR_(category, var) STR__(debug_enum, type)
    // #define STR(var) STR_(ENUM, var)
    // #define VAR source
    // #define ENUM SOURCE
    // const char* source_str =
    //     STR(API)
    //     STR(WINDOW_SYSTEM)
    //     STR(SHADER_COMPILER)
    //     STR(THIRD_PARTY)
    //     STR(APPLICATION)
    //     STR(OTHER)
    //     "Unknown";
    // #define ENUM TYPE
    // const char* type_str =
    //     STR(ERROR)
    //     STR(DEPRECATED_BEHAVIOR)
    //     STR(UNDEFINED_BEHAVIOR)
    //     STR(PORTABILITY)
    //     STR(PERFORMANCE)
    //     STR(MARKER)
    //     STR(PUSH_GROUP)
    //     STR(POP_GROUP)
    //     STR(OTHER)
    //     "Unknown";
    // #define ENUM SEVERITY
    // const char* severity_str =
    //     STR(HIGH)
    //     STR(MEDIUM)
    //     STR(LOW)
    //     STR(NOTIFICATION)
    //     "Unknown";
    
    fprintf(
        stderr,
        "GL debug: [source=%x type=%x severity=%x] %s\n",
        source,
        type,
        severity,
        message
    );
}


void GlSetup()
{
    // Initialize GLEW.
    glewInit();

    // Set debug message callback.
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(GlDebugMessageCallback, 0);
    
    glDebugMessageControl(
        GL_DONT_CARE, /*source*/
        GL_DONT_CARE, /*type*/
        GL_DONT_CARE, /*severity*/
        0,
        0,
        GL_TRUE
    );

    glDebugMessageControl(
        GL_DONT_CARE,
        GL_DONT_CARE,
        GL_DEBUG_SEVERITY_NOTIFICATION,
        0,
        0,
        GL_FALSE
    );
}


void GlDestroy()
{
}


void GlInfo()
{
    printf("GLEW:\n");
    printf("  Version: %s\n", glewGetString(GLEW_VERSION));

    printf("OpenGL:\n");
    printf("  Vendor:   %s\n", glGetString(GL_VENDOR));
    printf("  Renderer: %s\n", glGetString(GL_RENDERER));
    printf("  Version:  %s\n", glGetString(GL_VERSION));
    printf("  GLSL:     %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    GLint context_flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &context_flags);
    printf("  Context flags:\n");
    printf("    FORWARD_COMPATIBLE: %s\n", context_flags & GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT ? "TRUE" : "FALSE");
    printf("    DEBUG:              %s\n", context_flags & GL_CONTEXT_FLAG_DEBUG_BIT ? "TRUE" : "FALSE");
    printf("    ROBUST_ACCESS:      %s\n", context_flags & GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT ? "TRUE" : "FALSE");
    // printf("    NO_ERROR:           %s\n", context_flags & GL_CONTEXT_FLAG_NO_ERROR_BIT ? "TRUE" : "FALSE");
}
