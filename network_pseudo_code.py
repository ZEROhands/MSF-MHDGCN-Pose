class ModifiedResnet:
    Initialization:
        - `backbone`: **Main network** (uses `ResNet` to extract deep features)
        - `multi_scale_pooling`: **Multi-scale pooling module**, extracts features with different receptive fields
        - `dropout1, dropout2`: Dropout layers, prevent overfitting
        - `upsample1, upsample2, upsample3`: **Progressive upsampling**, restores resolution
        - `final_conv`: `1x1` convolution, adjusts channel dimensions

    Forward propagation (forward):
        1. **Extract base features**:
            - `f, global_f = backbone.extract_features(x)`
                - `x` shape: `(batch_size, 3, H, W)` (input RGB image)
                - `f = backbone(x)`: Processes through **convolution, batch normalization, ReLU**, extracting **multi-level CNN semantic features**  

        2. **Multi-scale feature processing**:
            - `p1 = multi_scale_pooling(f, scale=1)`: **Global feature pooling**
            - `p2 = multi_scale_pooling(f, scale=2)`: **Medium receptive field pooling**
            - `p3 = multi_scale_pooling(f, scale=3)`: **Local feature pooling**
            - `p4 = multi_scale_pooling(f, scale=6)`: **Smaller receptive field detail pooling**
            - `p = concat([p1, p2, p3, p4, f])`: **Merge multi-scale information**
            - `p = conv1x1(p)`: `1x1` convolution, **reduces channel dimensions, improving computational efficiency**
            - `p = dropout1(p)`: Apply Dropout to prevent overfitting

        3. **Progressive upsampling to restore spatial information**:
            - `p = upsample1(p) → dropout2(p)`: **First upsampling step**
            - `p = upsample2(p) → dropout2(p)`: **Second upsampling step**
            - `p = upsample3(p)`: **Final upsampling to target resolution**

        4. **Final feature output**:
            - `out = final_conv(p)`: Adjust channel dimensions using `1x1` convolution

    Output:
        - `out`: `(batch_size, C', H', W')` (final feature map)



def get_graph_feature_multihead(x, k=20, heads=8):
    Input:
        - `x`: Input point cloud features of shape `(batch_size, channels, num_points)`
        - `k`: Number of nearest neighbors to select
        - `heads`: Number of attention heads

    Processing:
        1. **Adjust input shape**:
            - `batch_size = x.shape[0]`
            - `num_points = x.shape[2]`
            - `x = x.reshape(batch_size, heads, -1, num_points)`: Split channels into multiple heads

        2. **Compute KNN neighborhood indices**:
            - `idx = knn_multihead(x, k)`: Compute K-nearest neighbors for each point
            - `idx_base = Generate batch dimension index offsets`
            - `idx = idx + idx_base`: Adapt indices to batch dimension
            - `idx = idx.reshape(-1)`: Flatten indices

        3. **Extract KNN neighborhood features**:
            - `feature = Extract nearest neighbor features from x using idx`
            - `feature = feature.reshape(batch_size, heads, num_points, k, num_dim)`

        4. **Compute center point information**:
            - `center = Compute center feature for each point`
            - `center_info = (x - center).reshape(...)`: Compute center offset information

        5. **Construct final feature representation**:
            - `x = x.reshape(batch_size, heads, num_points, 1, num_dim).repeat(1, 1, 1, k, 1)`
            - `feature = torch.cat([feature - x, x, center_info], dim=-1)`: Concatenate features
            - `feature = feature.reshape(batch_size, -1, num_points, k)`

    Output:
        - `feature`: `(batch_size, heads * num_dim * 3, num_points, k)`, containing multi-head local features



class MHDGCN:
    Initialization:
        - `emb_dims`: Feature dimension
        - `k`: Number of nearest neighbors in KNN
        - `heads`: Number of multi-heads
        - `dropout`: Dropout rate during training
        - `conv1, conv2, conv3, conv4`: Convolution layers for extracting features from local neighborhoods
        - `bn1, bn2, bn3, bn4`: Normalization layers to maintain numerical stability
        - Uses multi-head KNN to compute local neighborhoods and extract multi-scale features

    Forward propagation (forward):
        1. **Obtain KNN neighborhood features**:
            - `x1 = get_graph_feature_multihead(x, k, heads=1)`: Compute point cloud neighborhoods
            - `x1 = conv1(x1).max(dim=-1)`: Extract local features

        2. **Multi-head feature extraction**:
            - `x2 = get_graph_feature_multihead(x1, k, heads=8)`: Extract richer local relationships using multi-head KNN
            - `x2 = conv2(x2).max(dim=-1)`

        3. **Deep feature extraction**:
            - `x3 = get_graph_feature_multihead(x2, k, heads=8)`
            - `x3 = conv3(x3).max(dim=-1)`

            - `x4 = get_graph_feature_multihead(x3, k, heads=8)`
            - `x4 = conv4(x4).max(dim=-1)`

    Output:
        - `x3, x4`: Multi-scale point cloud features (low-level & high-level features)


class MFP: **MFP_Block2 from the paper**, extracts multi-scale features
    Initialization:
        - `branch1`: **Max Pooling**, extracts global important features
        - `branch2`: **Avg Pooling**, smooths features to reduce noise
        - `branch3_1`: **Small kernel feature extraction** (1x1 → 3x3 → 3x3)
        - `branch3_2`: **Large kernel feature extraction** (1x1 → 3x3)
        - `attention`: **SE-Block**, computes channel attention
        - `w`: **Weights for each branch**, shape (1 × 4)

    Forward propagation:
        1. **Compute different pooling branches**:
            - `b1 = max_pooling(input, kernel=3, stride=1)`
            - `b2 = avg_pooling(input, kernel=3, stride=1)`

        2. **Extract multi-scale features using convolution**:
            - `b3_1 = Conv1D(input, C/4) → Conv1D(3x3) → Conv1D(3x3)`
            - `b3_2 = Conv1D(input, 3C/4) → Conv1D(3x3)`

        3. **Enhance features using channel attention**:
            - `b3 = concat(b3_1, b3_2)`
            - `b3 = attention(b3)`

        4. **Compute normalized weights**:
            - `w1, w2, w3, w4 = softmax(w)`

        5. **Fuse all branch features**:
            - `output = w1 * b1 + w2 * b2 + w3 * b3 + w4 * input`

    Output:
        - `output`: **Fused feature map**



class Cross_attention:
    Initialization:
        - `M`: **Number of input branches, set to 2**
        - `global_pool`: **Global average pooling**
     
    Forward propagation:
      1. **Input two branch features `input1, input2`**:
            - `batch_size = input1.shape[0]`

        2. **Compute fused features**:
            - `U = input1 + input2`

        3. **Channel attention computation**:
            - `s = global_pool(U)`
            - `z = ReLU(Conv1D(s, d))`  # Dimension reduction
            - `a_b = Conv1D(z, C*M)`  # Dimension expansion
            - `a_b = reshape(a_b, (M, C))`
            - `weights = softmax(a_b)`

        4. **Compute final weighted features**:
            - `V1 = output1 * weights[0]`
            - `V2 = output2 * weights[1]`
            - `V = V1 + V2`

    Output:
        - `V`: **Cross-modal attention fused branch features, output fused features**




class PoseNet:
    Initialization:
        - `graph_net`: **MHDGCN**, used for point cloud feature extraction
        - `cnn`: **ModifiedResNet**, used for RGB feature extraction
        - `conv_r, conv_t, conv_c`: Convolution layers for predicting **rotation, translation, and confidence score**

    Forward propagation (forward):
        1. **RGB Feature Extraction**:
            - `emb = cnn(img)`: Extract **global RGB semantic features** through CNN
            - `emb = Select pixel features that match the point cloud`
        
        2. **Point Cloud Feature Extraction (Graph Convolution)**:
            - `x1, x2 = graph_net(x)`: Use **dynamic graph convolution network (MHDGCN)** for point cloud feature extraction
            - `x1`: Low-level point cloud features
            - `x2`: High-level point cloud features

        3. **Feature Fusion**:
            - **First feature fusion**:
                - `emb = ReLU(Conv1D(emb, 64))`: Low-level RGB features
                - `g = ReLU(Conv1D(g1, 64))`              
                - `feat1 = Cross_attention(concat(emb, g))`

            - **Second feature fusion**:
	- `emb = MFP(emb)`: Use **MFP** to extract further multi-scale features
                - `emb = ReLU(Conv1D(emb, 128))`
	- `g = MFP(x2)`: Use **MFP** to extract further multi-scale features
                - `g = ReLU(Conv1D(x2, 128))`
                - `feat2 = Cross_attention(concat(emb, g))`

            - **Global feature extraction**:
                - `x = ReLU(Conv1D(feat2, 512))`
                - `x = ReLU(Conv1D(x, 1024))`
                - `global_feat = global_max_pooling(x)`: **Max pooling to obtain global features**
                - `final_feat = concat(feat1, feat2, global_feat)`

        4. **Pose Regression**:
            - `rx = ReLU(conv_r(final_feat))`
            - `tx = ReLU(conv_t(final_feat))`
            - `cx = Sigmoid(conv_c(final_feat))`

        5. **Target class-specific output**:
            - `rx = select_target(rx, obj)`
            - `tx = select_target(tx, obj)`
            - `cx = select_target(cx, obj)`

        6. **Reshape and return**:
            - `rx = rx.reshape(batch_size, num_points, 4)`
            - `tx = tx.reshape(batch_size, num_points, 3)`
            - `cx = cx.reshape(batch_size, num_points, 1)`

    Output:
        - `rx`: **Rotation quaternion of the target object**
        - `tx`: **Translation vector of the target object**
        - `cx`: **Confidence score of the target object**



class PoseRefineNet:
    Initialization:
        - `graph_net`: **MHDGCN (Dynamic Graph Convolution Network)**, extracts point cloud features
        - **Fully connected layers** (`Linear`):
            - `conv1_r, conv2_r, conv3_r`: **Progressive refinement of rotation**
            - `conv1_t, conv2_t, conv3_t`: **Progressive refinement of translation**

    Forward propagation (forward):
        1. **Input `emb` (RGB features) & `x` (point cloud features)**
            - `batch_size = x.shape[0]`

        2. **Point Cloud Feature Extraction (Graph Convolution)**
            - `x1, x2 = graph_net(x)`: **MHDGCN low-level and high-level features**

        - **First Feature Fusion**:
            - `emb = ReLU(Conv1D(emb, 64))`
            - `g1 = ReLU(Conv1D(x2, 64))`
            - `feat1 = Cross_attention(concat(emb, g1))`

        - **Second Feature Fusion**:
            - `emb = MFP(emb)`: Uses **MFP** to further extract multi-scale features
            - `g2 = MFP(g2)`: Uses **MFP** to further extract multi-scale features
            - `emb = ReLU(Conv1D(emb, 128))`
            - `g2 = ReLU(Conv1D(g2, 128))`
            - `feat2 = Cross_attention(concat(g1, g2))`

        - **Global Feature Extraction**:
            - `x = ReLU(Conv1D(concat(feat1, feat2), 512))`
            - `x = ReLU(Conv1D(x, 1024))`
            - `global_feat = global_max_pooling(x)`: **Max pooling to obtain global features**

        4. **Rotation & Translation Refinement**
            - `rx = ReLU(conv1_r(global_feat))`
            - `tx = ReLU(conv1_t(global_feat))`
            - `rx = ReLU(conv2_r(rx))`
            - `tx = ReLU(conv2_t(tx))`
            - `rx = conv3_r(rx).reshape(batch_size, num_obj, 4)`
            - `tx = conv3_t(tx).reshape(batch_size, num_obj, 3)`

       5. **Select Target Class Output**:
            - `rx = select_target(rx, obj)`
            - `tx = select_target(tx, obj)`

       6. **Reshape and Return**:
            - `rx = rx.reshape(batch_size, 1, 4)`
            - `tx = tx.reshape(batch_size, 1, 3)`

    Output:
        - `out_rx`: **Refined rotation quaternion**
        - `out_tx`: **Refined translation vector**





