#include "lstruct.hpp"
#include "lamina.hpp"
#include <algorithm>

lStruct::lStruct() {
	size_t init_size = 16;
	buckets_.resize(init_size, nullptr);
}

// 初始化lStruct
lStruct::lStruct(const std::vector<std::pair<std::string, Value>>& vec) {
    // 初始化桶数组
    size_t init_size = 16;
    while (init_size < (vec.size() / load_factor_)) {
        init_size *= 2;
    }
    buckets_.resize(init_size, nullptr);

    // 插入所有初始键值对
    for (auto [name, val] : vec) {
        insert(name, val);
    }
}

lStruct::~lStruct() = default;

// 查找键对应的节点（返回nullptr表示未找到）
std::shared_ptr<Node> lStruct::find(const std::string& key) const {
    if (buckets_.empty()) return nullptr;   // 桶数组未初始化时直接返回

    const size_t hash = hash_string(key);
    const size_t bucket_idx = getBucketIndex(hash);

    // 遍历桶内链表查找（先比哈希，再比字符串）
    std::shared_ptr<Node> current = buckets_[bucket_idx];
    while (current != nullptr) {
        if (current->hash == hash && current->key == key) {
            return current; // 找到匹配节点
        }
        current = current->next;
    }

    return nullptr; // 未找到
}

// 插入或更新键值对
Value lStruct::insert(const std::string& key, Value& val) {
    if (buckets_.empty()) {
        buckets_.resize(16, nullptr);   // 初始化为16个桶
    }

    // 检查负载因子
    if (static_cast<float>(elem_count_) / buckets_.size() >= load_factor_) {
        this->resize();
    }

    const size_t hash = hash_string(key);
    const size_t bucket_idx = getBucketIndex(hash);

    // 是否已存在
    std::shared_ptr<Node> current = buckets_[bucket_idx];
    while (current != nullptr) {
        if (current->hash == hash && current->key == key) {
            current->value = std::move(val);
            return LAMINA_NULL;
        }
        current = current->next;
    }

    // 不存在则创建新节点
    const auto new_node = std::make_shared<Node>(key, std::move(val));
    new_node->next = buckets_[bucket_idx];  // 新节点的next指向原头节点
    buckets_[bucket_idx] = new_node;
    elem_count_++;
    return LAMINA_NULL;
}

std::vector<std::pair<std::string, Value>> lStruct::to_vector() const {
    std::vector<std::pair<std::string, Value>> vec;
    for (const std::shared_ptr<Node>& bucket_head : buckets_) {
        auto current = bucket_head;
        while (current != nullptr) {
            vec.emplace_back(current->key, current->value);
            current = current->next;
        }
    }
    return vec;
}


std::string lStruct::to_string() const {
    std::string text = "{\n";
    for (const std::shared_ptr<Node>& bucket_head : buckets_) {
        auto current = bucket_head;
        while (current != nullptr) {
            text += current->key + ": " + current->value.to_string();
            text += ",\n";
            current = current->next;
        }
    }
    text += "}";
    return text;
}


Value new_lstruct(const std::vector<std::pair<std::string, Value>>& vec) {
    const auto lstruct_ptr = std::make_shared<lStruct>(vec);
    return Value(lstruct_ptr);
}

Value getattr_raw(const std::shared_ptr<lStruct>& lstruct_, const std::string& attr_name) {
	auto res = lstruct_->find(attr_name);
    if (res == nullptr) {
        L_ERR("AttrError: struct hasn't attribute named " + attr_name);
        return LAMINA_NULL;
    }
    return res->value;
}

Value getattr(const std::vector<Value>& args) {
    const auto& lstruct_ = std::get<std::shared_ptr<lStruct>>(args[0].data);
    const auto& attr_name = std::get<std::string>(args[1].data);
    return getattr_raw(lstruct_, attr_name);
}

void setattr_raw(std::shared_ptr<lStruct> lstruct_, const std::string& attr_name, Value value) {
	lstruct_->insert(attr_name, value);
}

Value setattr(const std::vector<Value>& args) {
    std::shared_ptr<lStruct> lstruct_ = std::get<std::shared_ptr<lStruct>>(args[0].data);
    const auto& attr_name = std::get<std::string>(args[1].data);
    Value value = args[2];
	setattr_raw(lstruct_, attr_name, value);
    return LAMINA_NULL;
}

Value update(const std::vector<Value>& args) {
    std::shared_ptr<lStruct> lstruct_a = std::get<std::shared_ptr<lStruct>>(args[0].data);
    const auto& lstruct_b = std::get<std::shared_ptr<lStruct>>(args[1].data);
    auto vec = lstruct_b->to_vector();
    for (auto& [key, val]: vec) {
        lstruct_a->insert(key, val);
    }
    return LAMINA_NULL;
}

std::string lStruct_to_string(const std::shared_ptr<lStruct>& lstruct) {
    if (lstruct == nullptr) {
        return "{}";
    }
    return lstruct->to_string();
}
