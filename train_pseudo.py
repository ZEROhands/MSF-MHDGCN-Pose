# 1. **Import Libraries & Parse Arguments**
Initialization:
    - Parse command-line arguments (`argparse`)
    - Set **dataset path, batch size, learning rate, optimizer parameters**
    - Configure **GPU devices & distributed training**
    - Set **random seed** to ensure reproducibility

# 2. **Data Loading**
Select based on `opt.dataset`:
    - `PoseDataset_ycb` (YCB Video dataset)
    - `PoseDataset_linemod` (LineMOD dataset)

Create:
    - `dataloader` (Training set)
    - `testdataloader` (Test set)

# 3. **Model Initialization**
Create:
    - `PoseNet`: Predicts the initial **6D pose**
    - `PoseRefineNet`: **Refines** the predicted pose
    - Load `resume_posenet` & `resume_refinenet` pretrained weights (if available)
    - **Distributed training** (`DDP`)

# 4. **Optimizer & Training Parameters**
If **starting training from `PoseNet`**:
    - `optimizer = Adam(PoseNet.parameters(), lr=opt.lr)`

If **entering `PoseRefineNet` training phase**:
    - `opt.refine_start = True`
    - `optimizer = Adam(PoseRefineNet.parameters(), lr=opt.lr)`

lr_scheduler = CosineAnnealingLR(optimizer, T_max=32)  # Cosine annealing learning rate decay strategy

# 5. **Training Loop**
**Iterate `epoch` times**
    - `dataloader.sampler.set_epoch(epoch)`

    **Iterate over `dataloader` to train `PoseNet`**
        - Extract `points, R, choose, img, target, model_points, idx`
        - `pred_r, pred_t, pred_c, emb = PoseNet(img, points, R, choose, idx)`
        - `points = zhijiao(points, R)  # Convert to Cartesian coordinates`
        - `loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w)`

        **If `refine_start`**
            - `pred_r, pred_t = PoseRefineNet(emb, new_points, R, idx)`
            - `loss_refine, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)`
            - `loss_refine.backward()`
        **Else**
            - `loss.backward()`

        - `optimizer.step()` **Update weights**
        - **Save the model every 1000 iterations**

# 6. **Testing Phase**
`PoseNet.eval()`
Iterate over `testdataloader`:
    - `pred_r, pred_t, pred_c, emb = PoseNet(img, points, R, choose, idx)`
    - `points = zhijiao(points, R)`
    - `dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w)`

    **If `refine_start`**
        - `pred_r, pred_t = PoseRefineNet(emb, new_points, R, idx)`
        - `dis = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)`

    **Record error `dis` and compute the average error**
    **If `test_dis < best_test`, save `PoseNet` / `PoseRefineNet` weights**

# 7. **Learning Rate & Training Strategy Adjustment (After One Epoch)**
    - `lr_scheduler.step()`
    - If `test_dis < decay_margin`, reduce `lr`
    - If `test_dis < refine_margin`, start `PoseRefineNet` training
