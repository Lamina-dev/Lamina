#include "console_ui.hpp"
#include "repl_input.hpp"

int main(const int argc, const char* const argv[]) {
    int a = 1;// JUST A TEST
    try {
        enable_ansi_escape();           // 设置编码
        return argv_parser(argc, argv); // 主要执行这里

    } catch (const CtrlCException&) {
        std::cout << "\nProgram received Ctrl+C signal, exited safely." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nProgram exception occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nProgram encountered an unknown exception." << std::endl;
        return 1;
    }
}
