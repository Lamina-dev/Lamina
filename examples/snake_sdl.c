//
// snake_sdl.c : 小型辅助库, 供 examples/snake.lm 通过 `static "snake_sdl"` 加载。
// 作用: 语言层无法读取指针指向的内存(SDL_Event/SDL_FRect), 由 C 层代劳。
//
//   snake_key_pressed(scancode) : 当前帧该键是否按下 (内部 SDL_PumpEvents)
//   snake_quit_pending()        : 是否有 SDL_EVENT_QUIT 待处理
//   snake_fill_rect(ren,x,y,w,h): 构造 SDL_FRect 并填充
//
// 编译: gcc -O2 -fPIC -shared -o libsnake_sdl.so snake_sdl.c -lSDL3
//

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
