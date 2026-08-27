// C 辅助函数负责读取 SDL_Event/SDL_FRect 指针数据。
// 编译: gcc -O2 -fPIC -shared -o libsnake_sdl.so snake_sdl.c -lSDL3

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>

int snake_key_pressed(int scancode) {
    SDL_PumpEvents();
    const bool *keys = SDL_GetKeyboardState(NULL);
    if (keys == NULL) return 0;
    return keys[scancode] ? 1 : 0;
}

int snake_quit_pending(void) {
    SDL_PumpEvents();
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        if (ev.type == SDL_EVENT_QUIT) return 1;
    }
    return 0;
}

void snake_fill_rect(void *renderer, int x, int y, int w, int h) {
    const SDL_FRect r = { (float)x, (float)y, (float)w, (float)h };
    SDL_RenderFillRect((SDL_Renderer *)renderer, &r);
}
