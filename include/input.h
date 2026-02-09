#pragma once

#include <string>
#include <vector>


namespace makemore {

// Use this class to allocate heap memory once for the input names.
// Then provide efficient string views to this memory.
class SourceFile {
private:
    std::string content_;

public:
    SourceFile() = default;

    SourceFile(SourceFile&&) = default;
    SourceFile& operator=(SourceFile&&) = default;
    SourceFile(const SourceFile&) = delete;
    SourceFile& operator=(const SourceFile&) = delete;

    explicit SourceFile(const std::string& path);

    auto get_lines() const -> std::vector<std::string_view>;

    auto raw() const -> std::string_view;
};


} // namespace makemore
