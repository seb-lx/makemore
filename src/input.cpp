#include "input.h"

#include <string>
#include <fstream>
#include <sstream>


namespace makemore {


SourceFile::SourceFile(const std::string& path):
    content_{}
{
    std::ifstream f(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!f) throw std::runtime_error("File not found: " + path);

    auto file_size = f.tellg();
    content_.resize(static_cast<std::size_t>(file_size));

    f.seekg(0);
    f.read(content_.data(), file_size);
}

auto SourceFile::get_lines() const -> std::vector<std::string_view> 
{
    std::vector<std::string_view> lines;

    auto sv = std::string_view{ content_ };
    auto pos = std::size_t{ 0 };
    while ((pos = sv.find('\n')) != std::string_view::npos) {
        auto line = sv.substr(0, pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
            
        lines.push_back(line);

        sv.remove_prefix(pos + 1);
    }
    if (!sv.empty()) {
        if (sv.back() == '\r') sv.remove_suffix(1);
        lines.push_back(sv);
    }

    return lines;
}

auto SourceFile::raw() const -> std::string_view
{
    return content_;
}


} // namespace makemore
