//
// Created by meian on 2026/8/3.
//

#pragma once
#include <memory>
#include <vector>

#include "lmx.h"

namespace lmx::hir {

namespace ty {

enum class Kind {
    Int, Frac, Text, Array, TypeVar, Bool, Maybe
};
struct HirType;
using Type = std::shared_ptr<HirType>;
struct HirType {
    Kind kind;
protected:
    /*
     * 与 ast::TypePool 相同的保证：类型实例仅由 hir::ty::Pool 创建，
     * 确保相同 HIR 类型全局唯一。
     */
    explicit HirType(const Kind kind) noexcept : kind(kind) {}

public:
    [[nodiscard]] Kind get_kind() const noexcept { return kind; }

    virtual ~HirType() = default;

    [[nodiscard]] virtual std::string to_string() const noexcept = 0;

    [[nodiscard]] virtual bool equals(const Type&) const noexcept = 0;
};


struct IntType : HirType {
    friend class Pool;
private:
    explicit IntType() noexcept : HirType(Kind::Int) {}

public:
    ~IntType() override = default;

    [[nodiscard]] std::string to_string() const noexcept override {
        return "Type(builtin int)";
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (other.get() == this) return true;
        if (kind != other->kind) return false;
        return true;
    }
};
struct FracType : HirType {
    friend class Pool;
private:
    explicit FracType() noexcept : HirType(Kind::Frac) {}

public:
    ~FracType() override = default;

    [[nodiscard]] std::string to_string() const noexcept override {
        return "Type(builtin frac)";
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (other.get() == this) return true;
        if (kind != other->kind) return false;
        return true;
    }
};
struct TextType : HirType {
    friend class Pool;
private:
    explicit TextType() noexcept : HirType(Kind::Text) {}

public:
    ~TextType() override = default;

    [[nodiscard]] std::string to_string() const noexcept override {
        return "Type(builtin text)";
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (other.get() == this) return true;
        if (kind != other->kind) return false;
        return true;
    }
};
struct ArrayType : HirType {
    friend class Pool;

    Type data_ty;
private:
    explicit ArrayType(Type data_ty) noexcept : HirType(Kind::Array), data_ty(std::move(data_ty)) {}

public:
    ~ArrayType() override = default;

    [[nodiscard]] std::string to_string() const noexcept override {
        return "Type(Array[" + data_ty->to_string() + "])";
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (other.get() == this) return true;
        if (kind != other->kind) return false;
        const auto& other_real = reinterpret_cast<const ArrayType&>(*other);
        return data_ty->equals(other_real.data_ty);
    }
};
struct BoolType : HirType {
    friend class Pool;
private:
    explicit BoolType() noexcept : HirType(Kind::Bool) {}

public:
    ~BoolType() override = default;

    [[nodiscard]] std::string to_string() const noexcept override {
        return "Type(builtin bool)";
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (other.get() == this) return true;
        if (kind != other->kind) return false;
        return true;
    }
};
struct TypeVar : HirType {
    friend class Pool;

    std::string name;
private:
    explicit TypeVar(std::string name) noexcept : HirType(Kind::TypeVar), name(std::move(name)) {}

public:
    ~TypeVar() override = default;

    [[nodiscard]] std::string to_string() const noexcept override {
        return "Type(generic " + name + ")";
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override = 0;
};
struct MaybeType : HirType {
    friend class Pool;

    Type data_ty;
private:
    explicit MaybeType(Type data_ty) noexcept : HirType(Kind::Maybe), data_ty(std::move(data_ty)) {}

public:
    ~MaybeType() override = default;
    [[nodiscard]] std::string to_string() const noexcept override {
        return "Type(" + data_ty->to_string() + "?)";
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (other.get() == this) return true;
        if (kind != other->kind) return false;
        const auto& other_real = reinterpret_cast<const MaybeType&>(*other);
        return data_ty->equals(other_real.data_ty);
    }
};




class Pool {
    std::vector<Type> types{};

    template<typename T, const Kind kind>
    [[nodiscard]] Type new_easy_ty() noexcept {
        for (auto& i : types)
            if (i->get_kind() == kind) return i;

        types.push_back(std::shared_ptr<HirType>(new T()));
        return types.back();
    }
public:
    explicit Pool() noexcept {
        types.push_back(std::shared_ptr<HirType>(new IntType()));
        types.push_back(std::shared_ptr<HirType>(new FracType()));
        types.push_back(std::shared_ptr<HirType>(new TextType()));
        types.push_back(std::shared_ptr<HirType>(new BoolType()));
    }
    [[nodiscard]] Type new_array_ty(const Type& data_ty) noexcept {
        for (const auto& i : types)
            if (i->get_kind() == Kind::Array)
                if (reinterpret_cast<const ArrayType&>(*i).data_ty->equals(data_ty))
                    return i;

        types.push_back(std::shared_ptr<HirType>(new ArrayType(data_ty)));
        return types.back();
    }
    [[nodiscard]] Type new_int_ty() noexcept {
        return new_easy_ty<IntType, Kind::Int>();
    }
    [[nodiscard]] Type new_frac_ty() noexcept {
        return new_easy_ty<FracType, Kind::Frac>();
    }
    [[nodiscard]] Type new_text_ty() noexcept {
        return new_easy_ty<TextType, Kind::Text>();
    }
    [[nodiscard]] Type new_bool_ty() noexcept {
        return new_easy_ty<BoolType, Kind::Bool>();
    }
    [[nodiscard]] Type new_maybe_ty(const Type& data_ty) noexcept {
        for (const auto& i : types)
            if (i->get_kind() == Kind::Maybe)
                if (reinterpret_cast<const MaybeType&>(*i).data_ty->equals(data_ty))
                    return i;

        types.push_back(std::shared_ptr<HirType>(new MaybeType(data_ty)));
        return types.back();
    }
};


}

struct HirModule {

};

struct HirSymTable {

};

struct HirNode {

};

}
