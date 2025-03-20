class ModifiedResnet:
    初始化:
        - `backbone`: **主干网络**（采用 `ResNet` 提取深层特征）
        - `multi_scale_pooling`: **多尺度池化模块**，用于提取不同感受野的特征
        - `dropout1, dropout2`: Dropout 层，防止过拟合
        - `upsample1, upsample2, upsample3`: **逐步上采样**，恢复分辨率
        - `final_conv`: `1x1` 卷积层，调整通道数

    前向传播 (forward):
        1. **提取基础特征**:
            - `f, global_f = backbone.extract_features(x)`
                - `x` 形状: `(batch_size, 3, H, W)`（输入 RGB 图像）
                - `f = backbone(x)`: 经过 **卷积层、批归一化、ReLU** 组合，提取 **多层级 CNN 语义特征**  

        2. **多尺度特征处理**:
            - `p1 = multi_scale_pooling(f, scale=1)`: **全局特征池化**
            - `p2 = multi_scale_pooling(f, scale=2)`: **中等感受野池化**
            - `p3 = multi_scale_pooling(f, scale=3)`: **局部特征池化**
            - `p4 = multi_scale_pooling(f, scale=6)`: **更小感受野的细节池化**
            - `p = concat([p1, p2, p3, p4, f])`: **融合多尺度信息**
            - `p = conv1x1(p)`: `1x1` 卷积层，**降低通道数，提高计算效率**
            - `p = dropout1(p)`: 适当 Dropout，防止过拟合

        3. **逐步上采样恢复空间信息**:
            - `p = upsample1(p) → dropout2(p)`: **第一步上采样**
            - `p = upsample2(p) → dropout2(p)`: **第二步上采样**
            - `p = upsample3(p)`: **最终恢复到合适分辨率**

        4. **最终特征输出**:
            - `out = final_conv(p)`: 通过 `1x1` 卷积调整通道数

    输出:
        - `out`: `(batch_size, C', H', W')`（最终特征图）


def get_graph_feature_multihead(x, k=20, heads=8):
    输入:
        - `x`: 形状为 `(batch_size, channels, num_points)` 的输入点云特征
        - `k`: 选择 K 个最近邻
        - `heads`: 设定的多头数量

    处理:
        1. **调整输入形状**:
            - `batch_size = x.shape[0]`
            - `num_points = x.shape[2]`
            - `x = x.reshape(batch_size, heads, -1, num_points)`: 将通道维度划分为多个头

        2. **计算 KNN 邻域索引**:
            - `idx = knn_multihead(x, k)`: 计算每个点的 K 近邻索引
            - `idx_base = 生成 batch 维度索引偏移量`
            - `idx = idx + idx_base`: 适配索引到 batch 维度
            - `idx = idx.reshape(-1)`: 展平索引

        3. **获取 KNN 邻域特征**:
            - `feature = 从 x 中根据 idx 提取最近邻特征`
            - `feature = feature.reshape(batch_size, heads, num_points, k, num_dim)`

        4. **计算中心点信息**:
            - `center = 计算每个点的中心点特征`
            - `center_info = (x - center).reshape(...)`: 计算中心偏移信息

        5. **构造最终特征**:
            - `x = x.reshape(batch_size, heads, num_points, 1, num_dim).repeat(1, 1, 1, k, 1)`
            - `feature = torch.cat([feature - x, x, center_info], dim=-1)`: 组合特征
            - `feature = feature.reshape(batch_size, -1, num_points, k)`

    输出:
        - `feature`: `(batch_size, heads * num_dim * 3, num_points, k)`，包含多头局部特征


class MHDGCN:
    初始化:
        - `emb_dims`: 特征维度
        - `k`: KNN 近邻点数
        - `heads`: 设定的多头数量
        - `dropout`: 训练时的 dropout 率
        - `conv1, conv2, conv3, conv4`: 卷积层，用于从局部邻域提取特征
        - `bn1, bn2, bn3, bn4`: 归一化层，保持数值稳定
        - 采用多头 KNN 计算局部邻域，并提取多尺度特征

    前向传播 (forward):
        1. **获取 KNN 邻域特征**:
            - `x1 = get_graph_feature_multihead(x, k,heads=1)`: 计算点云邻域
            - `x1 = conv1(x1).max(dim=-1)`: 提取局部特征

        2. **多头特征提取**:
            - `x2 = get_graph_feature_multihead(x1, k,heads=8)`: 通过多头 KNN 提取更丰富的局部关系
            - `x2 = conv2(x2).max(dim=-1)`

        3. **深层特征提取**:
            - `x3 = get_graph_feature_multihead(x2, k,heads=8)`
            - `x3 = conv3(x3).max(dim=-1)`

            - `x4 = get_graph_feature_multihead(x3, k,heads=8)`
            - `x4 = conv4(x4).max(dim=-1)`

    输出:
        - `x3, x4`: 多尺度点云特征（低级 & 高级特征）

class MFP: **论文中的MFP_Block2**，提取多尺度特征
    初始化:
        - `branch1`: **最大池化（Max Pooling）**，提取全局重要特征
        - `branch2`: **平均池化（Avg Pooling）**，平滑特征以减少噪声
        - `branch3_1`: **小卷积核特征提取**（1x1 → 3x3 → 3x3）
        - `branch3_2`: **大卷积核特征提取**（1x1 → 3x3）
        - `attention`: **SE-Block**，计算通道注意力
        - `w`: **各分支权重**形状为（1 x 4）

    前向传播:
        1. **分支计算不同池化方式**:
            - `b1 = max_pooling(input, kernel=3, stride=1)`
            - `b2 = avg_pooling(input, kernel=3, stride=1)`

        2. **卷积提取多尺度特征**:
            - `b3_1 = Conv1D(input, C/4) → Conv1D(3x3) → Conv1D(3x3)`
            - `b3_2 = Conv1D(input, 3C/4) → Conv1D(3x3)`

        3. **通道注意力增强**:
            - `b3 = concat(b3_1, b3_2)`
            - `b3 = attention(b3)`

        4. **计算归一化权重**:
            - `w1, w2, w3, w4 = softmax(w)`

        5. **融合所有分支特征**:
            - `output = w1 * b1 + w2 * b2 + w3 * b3 + w4 * input`

    输出:
        - `output`: **融合后的特征图**


class Cross_attention:
    初始化:
        - `M`: **输入分支数量,设置为2**
        - `global_pool`: **全局平均池化**
     
    前向传播:
      1. **输入两个分支的特征 `input1, input2`**:
            - `batch_size = input1.shape[0]`

        2. **计算融合特征**:
            - `U = iutput1 + iutput2`

        3. **通道注意力计算**:
            - `s = global_pool(U)`
            - `z = ReLU(Conv1D(s, d))`  # 降维
            - `a_b = Conv1D(z, C*M)`  # 升维
            - `a_b = reshape(a_b, (M, C))`
            - `weights = softmax(a_b)`

        4. **计算最终加权特征**:
            - `V1 = output1 * weights[0]`
            - `V2 = output2 * weights[1]`
            - `V = V1 + V2`

    输出:
        - `V`: **跨模态注意力融合分支特征，输出融合特征**



class PoseNet:
    初始化:
        - `graph_net`: **MHDGCN**，用于点云特征提取
        - `cnn`: **ModifiedResNet**，用于 RGB 特征提取
        - `conv_r, conv_t, conv_c`: 预测 **旋转、平移、置信度** 的卷积层

    前向传播 (forward):
        1. **RGB 特征提取**:
            - `emb = cnn(img)`: 通过 CNN 提取 **全局 RGB 语义特征**
            - `emb = 选择与点云匹配的像素特征`
        
        2. **点云特征提取（Graph Convolution）**:
            - `x1, x2 = graph_net(x)`: 采用 **动态图卷积网络（MHDGCN）** 进行点云特征提取
            - `x1`: 低级点云特征
            - `x2`: 高级点云特征

        3. **特征融合**:
            - **第一次特征融合**：
                - `emb = ReLU(Conv1D(emb, 64))`: 低级 RGB 特征
                - `g = ReLU(Conv1D(g1, 64))`              
                - `feat1 = Cross_attention(concat(emb, g))`

            - **第二次特征融合**：
	- `emb = MFP(emb)`: 采用 **MFP ** 进一步提取多尺度特征
                - `emb = ReLU(Conv1D(emb, 128))`
	- `g = MFP(x2)`: 采用 **MFP ** 进一步提取多尺度特征
                - `g = ReLU(Conv1D(x2, 128))`
                - `feat2 = Cross_attention(concat(emb, g))`

            - **全局特征提取**：
                - `x = ReLU(Conv1D(feat2, 512))`
                - `x = ReLU(Conv1D(x, 1024))`
                - `global_feat = global_max_pooling(x)`: **最大池化获取全局特征**
                - `final_feat = concat(feat1, feat2, global_feat)`

        4. **位姿回归（Pose Regression）**:
            - `rx = ReLU(conv_r(final_feat))`
            - `tx = ReLU(conv_t(final_feat))`
            - `cx = Sigmoid(conv_c(final_feat))`

        5. **目标类别相关输出**:
            - `rx = select_target(rx, obj)`
            - `tx = select_target(tx, obj)`
            - `cx = select_target(cx, obj)`

        6. **调整形状并返回**:
            - `rx = rx.reshape(batch_size, num_points, 4)`
            - `tx = tx.reshape(batch_size, num_points, 3)`
            - `cx = cx.reshape(batch_size, num_points, 1)`

    输出:
        - `rx`: **目标物体的旋转四元数**
        - `tx`: **目标物体的平移向量**
        - `cx`: **目标物体的置信度**


class PoseRefineNet:
    初始化:
        - `graph_net`: **MHDGCN（动态图卷积网络）**，提取点云特征
        - **全连接层**（`Linear`）：
            - `conv1_r, conv2_r, conv3_r`: **逐步优化旋转**
            - `conv1_t, conv2_t, conv3_t`: **逐步优化平移**

    前向传播 (forward):
        1. **输入 `emb`（RGB 特征）& `x`（点云特征）**
            - `batch_size = x.shape[0]`

        
        2. **点云特征提取（Graph Convolution）**
            - `x1, x2 = graph_net(x)`: **MHDGCN 低级和高级特征**

         - **第一次特征融合**：
                - `emb = ReLU(Conv1D(emb, 64))`
                - `g1 = ReLU(Conv1D(x2, 64))`
                - `feat1 = Cross_attention(concat(emb, g1))`

            - **第二次特征融合**：
                - `emb = MFP(emb)`: 采用 **MFP** 进一步提取多尺度特征
                - `g2 = MFP(g2)`: 采用 **MFP** 进一步提取多尺度特征
                - `emb = ReLU(Conv1D(emb, 128))`
                - `g2 = ReLU(Conv1D(g2, 128))`
                - `feat2 = Cross_attention(concat(g1, g2))`

            - **全局特征提取**：
                - `x = ReLU(Conv1D(concat(feat1,feat2), 512))`
                - `x = ReLU(Conv1D(x, 1024))`
                - `global_feat = global_max_pooling(x)`: **最大池化获取全局特征**

        4. **旋转 & 平移优化**
            - `rx = ReLU(conv1_r(global_feat))`
            - `tx = ReLU(conv1_t(global_feat))`
            - `rx = ReLU(conv2_r(rx))`
            - `tx = ReLU(conv2_t(tx))`
            - `rx = conv3_r(rx).reshape(batch_size, num_obj, 4)`
            - `tx = conv3_t(tx).reshape(batch_size, num_obj, 3)`

       5. **选择目标类别输出**:
            - `rx = select_target(rx, obj)`
            - `tx = select_target(tx, obj)`

       6. **调整形状并返回**:
            - `rx = rx.reshape(batch_size, 1, 4)`
            - `tx = tx.reshape(batch_size, 1, 3)`

    输出:
        - `out_rx`: **优化后的旋转四元数**
        - `out_tx`: **优化后的平移向量**




