# 1. **导入库 & 解析参数**
初始化:
    - 解析命令行参数 (`argparse`)
    - 设置 **数据集路径、批量大小、学习率、优化器参数**
    - 设定 **GPU 设备 & 分布式训练**
    - 设定 **随机种子** 以保证可复现性

# 2. **数据加载**
根据 `opt.dataset` 选择:
    - `PoseDataset_ycb` （YCB 视频数据集）
    - `PoseDataset_linemod` （LineMOD 数据集）

创建:
    - `dataloader` （训练集）
    - `testdataloader` （测试集）

# 3. **模型初始化**
创建:
    - `PoseNet`: 预测初始 **6D 位姿**
    - `PoseRefineNet`: **优化** 预测的位姿
    - 载入 `resume_posenet` & `resume_refinenet` 预训练权重（若有）
    - **分布式训练** (`DDP`)

# 4. **优化器 & 训练参数**
如果 **从 `PoseNet` 开始训练**:
    - `optimizer = Adam(PoseNet.parameters(), lr=opt.lr)`

如果 **进入 `PoseRefineNet` 训练阶段**:
    - `opt.refine_start = True`
    - `optimizer = Adam(PoseRefineNet.parameters(), lr=opt.lr)`

lr_scheduler = CosineAnnealingLR(optimizer, T_max=32)  #余弦退火的学习率衰减策略

# 5. **训练循环**
**循环 `epoch` 次**
    - `dataloader.sampler.set_epoch(epoch)`

    **遍历 `dataloader` 训练 `PoseNet`**
        - 取出 `points, R, choose, img, target, model_points, idx`
        - `pred_r, pred_t, pred_c, emb = PoseNet(img, points, R, choose, idx)`
        - `points = zhijiao(points, R)  # 变换为直角坐标`
        - `loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w)`

        **若 `refine_start`**
            - `pred_r, pred_t = PoseRefineNet(emb, new_points, R, idx)`
            - `loss_refine, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)`
            - `loss_refine.backward()`
        **否则**
            - `loss.backward()`

        - `optimizer.step()` **更新权重**
        - **每 1000 次保存一次模型**

# 6. **测试阶段**
`PoseNet.eval()`
遍历 `testdataloader`:
    - `pred_r, pred_t, pred_c, emb = PoseNet(img, points, R, choose, idx)`
    - `points = zhijiao(points, R)`
    - `dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w)`

    **若 `refine_start`**
        - `pred_r, pred_t = PoseRefineNet(emb, new_points, R, idx)`
        - `dis = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)`

    **记录误差 `dis` 并计算平均误差**
    **若 `test_dis < best_test`，保存 `PoseNet` / `PoseRefineNet` 权重**

# 7. **学习率 & 训练策略调整（一个epoch结束后）**
    - `lr_scheduler.step()`
    - 若 `test_dis < decay_margin`，降低 `lr`
    - 若 `test_dis < refine_margin`，开始 `PoseRefineNet` 训练
