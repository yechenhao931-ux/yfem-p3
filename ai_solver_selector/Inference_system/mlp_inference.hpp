#pragma once
/**
 * mlp_inference.hpp
 * =================
 * 纯 C++17 MLP 前向推理，精确复现 train_torch.py 导出的残差 MLP。
 *
 * 不依赖任何机器学习库，仅用标准 C++ + nlohmann/json。
 *
 * 网络结构（与 Python 端完全对应）：
 *   Input(19)
 *     └─ ResBlock(in→256): Linear → BN(eval) → ReLU → (+ proj(in→256))
 *     └─ ResBlock(256→128): Linear → BN(eval) → ReLU → (+ proj(256→128))
 *     └─ ResBlock(128→64):  Linear → BN(eval) → ReLU → (+ 64 identity)
 *     └─ Head: Linear(64→8)
 *     └─ Softmax
 *
 * 使用方式：
 *   #include "mlp_inference.hpp"
 *   MlpSolverModel model;
 *   model.LoadFromFile("models/solver_model_torch.json");
 *   auto proba = model.Predict(feature_vector);   // std::vector<double>, size=19
 *   int  best  = model.PredictClass(feature_vector);
 */

#ifndef MLP_INFERENCE_HPP
#define MLP_INFERENCE_HPP

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <fstream>

#include "json.hpp"
using json = nlohmann::json;

// ======================================================================
//  线性代数原语（内联，不引入额外依赖）
// ======================================================================
namespace mlp_detail {

using Vec = std::vector<float>;
using Mat = std::vector<Vec>;  // [rows][cols]

/// y = W * x + b,  W: [out × in]
inline Vec linear(const Mat& W, const Vec& b, const Vec& x) {
    size_t out = W.size(), in = x.size();
    Vec y(out, 0.0f);
    for (size_t i = 0; i < out; ++i) {
        float acc = b[i];
        const Vec& row = W[i];
        for (size_t j = 0; j < in; ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
    return y;
}

/// Batch Normalization（推理模式，γ/β/running_mean/running_var）
inline Vec batch_norm(const Vec& x,
                      const Vec& gamma,
                      const Vec& beta,
                      const Vec& running_mean,
                      const Vec& running_var,
                      float eps = 1e-5f) {
    size_t n = x.size();
    Vec y(n);
    for (size_t i = 0; i < n; ++i) {
        float inv_std = 1.0f / std::sqrt(running_var[i] + eps);
        y[i] = gamma[i] * (x[i] - running_mean[i]) * inv_std + beta[i];
    }
    return y;
}

/// ReLU
inline Vec relu(Vec x) {
    for (auto& v : x) v = std::max(0.0f, v);
    return x;
}

/// Softmax（数值稳定版）
inline Vec softmax(Vec x) {
    float mx = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (auto& v : x) { v = std::exp(v - mx); sum += v; }
    for (auto& v : x) v /= sum;
    return x;
}

/// 元素加法
inline Vec add(const Vec& a, const Vec& b) {
    Vec c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
    return c;
}

/// 归一化特征（统一公式，支持 StandardScaler / MinMaxScaler / RobustScaler）
///
/// 公式：x_scaled = clip((x - mean) / scale, clip_lo, clip_hi)
///
/// - StandardScaler:  mean = μ,        scale = σ,       clip=[−∞, +∞]
/// - MinMaxScaler:    mean = data_min, scale = range,    clip=[0, 1]
/// - RobustScaler:    mean = median,   scale = IQR,      clip=[−∞, +∞]
///
/// 传入 clip_enabled=false 时完全跳过 clip 分支（稍快）。
inline Vec scale(const std::vector<double>& raw,
                 const Vec& mean,
                 const Vec& scale_v,
                 bool  clip_enabled = false,
                 float clip_lo = 0.0f,
                 float clip_hi = 1.0f)
{
    Vec xs(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        float s = (scale_v[i] > 1e-12f) ? scale_v[i] : 1.0f;
        float v = static_cast<float>((raw[i] - mean[i]) / s);
        if (clip_enabled) {
            if      (v < clip_lo) v = clip_lo;
            else if (v > clip_hi) v = clip_hi;
        }
        xs[i] = v;
    }
    return xs;
}

} // namespace mlp_detail


// ======================================================================
//  ResBlock 参数结构
// ======================================================================
struct ResBlockParams {
    int in_dim  = 0;
    int out_dim = 0;

    mlp_detail::Mat fc_weight;   // [out_dim × in_dim]
    mlp_detail::Vec fc_bias;     // [out_dim]

    mlp_detail::Vec bn_weight;        // γ  [out_dim]
    mlp_detail::Vec bn_bias;          // β  [out_dim]
    mlp_detail::Vec bn_running_mean;  // μ  [out_dim]
    mlp_detail::Vec bn_running_var;   // σ² [out_dim]
    float           bn_eps = 1e-5f;

    bool            has_proj = false;
    mlp_detail::Mat proj_weight;  // [out_dim × in_dim]（in≠out 时存在）

    /// 前向传播：x → ReLU(BN(Wx + b)) + proj(x)
    mlp_detail::Vec Forward(const mlp_detail::Vec& x) const {
        using namespace mlp_detail;

        // 主路径
        Vec h = linear(fc_weight, fc_bias, x);
        h     = batch_norm(h, bn_weight, bn_bias,
                           bn_running_mean, bn_running_var, bn_eps);
        h     = relu(h);

        // 残差路径
        Vec skip = has_proj ? linear(proj_weight,
                                      Vec(out_dim, 0.0f),  // proj 无 bias
                                      x)
                            : x;

        return add(h, skip);
    }
};


// ======================================================================
//  MlpSolverModel — 完整推理模型
// ======================================================================
class MlpSolverModel {
public:
    int                      n_features = 0;
    int                      n_classes  = 0;
    std::vector<std::string> feature_names;
    std::vector<std::string> class_names;

    // ── 加载 ──────────────────────────────────────────────
    void LoadFromFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("无法打开模型文件：" + path);

        json j;
        f >> j;

        if (j.at("model_type").get<std::string>() != "MLP_ResNet")
            throw std::runtime_error("模型类型不匹配，期望 MLP_ResNet");

        n_features   = j.at("n_features").get<int>();
        n_classes    = j.at("n_classes").get<int>();
        feature_names = j.at("feature_names").get<std::vector<std::string>>();
        class_names   = j.at("class_names").get<std::vector<std::string>>();

        // Scaler（兼容新旧字段）
        const auto& js = j.at("scaler");
        scaler_mean_  = js.at("mean").get<std::vector<float>>();
        scaler_scale_ = js.at("scale").get<std::vector<float>>();

        // scaler.kind: "standard" | "minmax" | "robust" (可选，旧模型无此字段)
        scaler_kind_ = js.contains("kind")
                        ? js.at("kind").get<std::string>()
                        : std::string("standard");

        // clip_lo / clip_hi: MinMax 训练时为 [0, 1]，其他为 null
        clip_enabled_ = false;
        if (js.contains("clip_lo") && js.contains("clip_hi") &&
            !js.at("clip_lo").is_null() && !js.at("clip_hi").is_null()) {
            clip_lo_      = (float)js.at("clip_lo").get<double>();
            clip_hi_      = (float)js.at("clip_hi").get<double>();
            clip_enabled_ = true;
        }

        // ResBlocks
        blocks_.clear();
        for (const auto& jblock : j.at("layers")) {
            ResBlockParams b;
            b.in_dim  = jblock.at("in_dim").get<int>();
            b.out_dim = jblock.at("out_dim").get<int>();

            b.fc_weight       = jblock.at("fc_weight").get<mlp_detail::Mat>();
            b.fc_bias         = jblock.at("fc_bias").get<mlp_detail::Vec>();
            b.bn_weight       = jblock.at("bn_weight").get<mlp_detail::Vec>();
            b.bn_bias         = jblock.at("bn_bias").get<mlp_detail::Vec>();
            b.bn_running_mean = jblock.at("bn_running_mean").get<mlp_detail::Vec>();
            b.bn_running_var  = jblock.at("bn_running_var").get<mlp_detail::Vec>();
            b.bn_eps          = (float)jblock.at("bn_eps").get<double>();

            if (jblock.contains("proj_weight")) {
                b.has_proj    = true;
                b.proj_weight = jblock.at("proj_weight").get<mlp_detail::Mat>();
            }
            blocks_.push_back(std::move(b));
        }

        // Head
        head_weight_ = j.at("head").at("weight").get<mlp_detail::Mat>();
        head_bias_   = j.at("head").at("bias").get<mlp_detail::Vec>();
    }

    // ── 前向推理 ──────────────────────────────────────────

    /**
     * @brief 给定原始（未标准化）特征向量，返回各类别 softmax 概率。
     * @param raw_features  长度 == n_features 的特征向量
     * @return              长度 == n_classes 的概率向量
     */
    std::vector<float> Predict(const std::vector<double>& raw_features) const {
        using namespace mlp_detail;

        if ((int)raw_features.size() != n_features)
            throw std::runtime_error(
                "特征维度不匹配：期望 " + std::to_string(n_features) +
                "，实际 " + std::to_string(raw_features.size()));

        // 1. 归一化（支持 standard / minmax / robust）
        Vec x = scale(raw_features, scaler_mean_, scaler_scale_,
                      clip_enabled_, clip_lo_, clip_hi_);

        // 2. 逐 ResBlock 前向
        for (const auto& block : blocks_)
            x = block.Forward(x);

        // 3. 线性头
        x = linear(head_weight_, head_bias_, x);

        // 4. Softmax
        x = softmax(x);
        return x;
    }

    /**
     * @brief 返回最优类别索引（0-based）。
     */
    int PredictClass(const std::vector<double>& raw_features) const {
        auto proba = Predict(raw_features);
        return (int)(std::max_element(proba.begin(), proba.end()) - proba.begin());
    }

    /**
     * @brief 返回最优类别名称。
     */
    const std::string& PredictClassName(const std::vector<double>& raw_features) const {
        int cls = PredictClass(raw_features);
        return class_names.at(cls);
    }

    /**
     * @brief 打印各类别置信度。
     */
    void PrintPrediction(const std::vector<double>& raw_features) const {
        auto proba = Predict(raw_features);
        int  best  = (int)(std::max_element(proba.begin(), proba.end()) - proba.begin());

        std::printf("─── MLP 求解器预测 ─────────────────────────\n");
        for (int c = 0; c < n_classes; ++c) {
            bool is_best = (c == best);
            std::printf("  %s%-20s  %.1f%%\033[0m\n",
                        is_best ? "\033[1;32m" : "",
                        class_names[c].c_str(),
                        proba[c] * 100.0f);
        }
        std::printf("  → 最优：\033[1;33m%s\033[0m  (置信度 %.1f%%)\n",
                    class_names[best].c_str(), proba[best] * 100.0f);
        std::printf("────────────────────────────────────────────\n");
    }

    bool IsLoaded() const { return !blocks_.empty(); }

    /// 返回 scaler 类型字符串 ("standard" | "minmax" | "robust")
    const std::string& ScalerKind() const { return scaler_kind_; }
    bool  ClipEnabled() const { return clip_enabled_; }
    float ClipLo()      const { return clip_lo_; }
    float ClipHi()      const { return clip_hi_; }

private:
    std::vector<ResBlockParams> blocks_;
    mlp_detail::Mat             head_weight_;
    mlp_detail::Vec             head_bias_;
    mlp_detail::Vec             scaler_mean_;
    mlp_detail::Vec             scaler_scale_;
    std::string                 scaler_kind_ = "standard";
    bool                        clip_enabled_ = false;
    float                       clip_lo_ = 0.0f;
    float                       clip_hi_ = 1.0f;
};

#endif // MLP_INFERENCE_HPP