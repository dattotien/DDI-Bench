# Data layout

Mỗi dataset có thư mục riêng — không dataset nào ghi đè dataset nào nữa.
Mọi thông số (`num_ent`, `num_rel`, `task`, tên file) khai báo tại
[`../dataset_registry.py`](../dataset_registry.py).

```
data/
  <dataset>_cluster/      train.txt, valid_S{0,1,2}.txt, test_S{0,1,2}.txt
  <dataset>_random/       (idem)
  initial/<dataset>/      DB_molecular_feats.pkl, id2smiles.json, relations_2hop.txt
```

Chọn dataset khi chạy: `python main.py --model MLP --dataset mecddi --dataset_type cluster`

Kiểm tra data trước khi train: `python check_data.py` (hoặc `python check_data.py mecddi cluster`).

## Tình trạng hiện tại

| dataset    | num_ent | num_rel | task       | eval      | cluster | random | initial/                                       |
|------------|---------|---------|------------|-----------|---------|--------|------------------------------------------------|
| `drugbank` | 1710    | 86      | multiclass | acc/F1/κ  | ✅      | ✅     | feats + id2smiles đủ 1710 drug + relations_2hop |
| `mecddi`   | 1567    | 103     | multiclass | acc/F1/κ  | ✅      | ❌     | feats + id2smiles đủ 1567 drug, **không có** relations_2hop |
| `mudi`     | 1295    | 4       | multiclass | có hướng  | ✅      | ❌     | feats + id2smiles đủ 1295 drug, **không có** relations_2hop |
| `twosides` | 645     | 209     | multilabel | PR/ROC-AUC| ✅      | ✅     | feats + cid2id + cid2smiles + relations_2hop   |

Cả bốn dataset đều đủ file để chạy, trừ `relations_2hop.txt` của MecDDI và MUDI
(nên Decagon / TIGER chỉ chạy được trên DrugBank và TWOSIDES).

`data/initial/drugbank/` nặng 41 MB (feature pickle 17 MB + `relations_2hop.txt`
25 MB); nếu cần dựng lại từ git thì
`git checkout 197db0b -- DDI_Ben/DDI_Ben/data/initial/drugbank`.

## MUDI

`data/mudi_cluster/` được sinh ra từ `data/mudi_raw/` bằng
[`../prepare_mudi.py`](../prepare_mudi.py):

```
python prepare_mudi.py            # dry run
python prepare_mudi.py --write    # ghi thật (bản cũ -> *.bak)
```

| split | nguồn | dòng |
|-------|-------|------|
| `train` | `MUDIv2_train.csv` | 346 859 |
| `valid_S0` / `valid_S1` / `valid_S2` | `MUDIv2_val.csv` (**ba bản copy y hệt**) | 113 886 mỗi file |
| `test_S0` / `test_S1` / `test_S2` | `test_S{0,1,2}.csv` | 130 298 / 174 886 / 38 220 |

MUDI chỉ có một tập val, còn `trainer.py` chọn model riêng cho từng `valid_S*`
ứng với từng `test_S*`, nên val được duplicate ba lần cho đủ tên file.

- **Nhãn** lấy từ cột `Pharmacodynamics`, đúng 4 lớp
  `No Interaction=0, Synergism=1, Antagonism=2, New Effect=3`. Cột
  `Pharmacokinetics` và `Adverse Effects` chưa dùng.
- **Node id** đánh theo thứ tự DrugBank ID tăng dần trên toàn bộ 1295 drug
  (union của mọi split), lưu lại ở `initial/mudi/node2drugbank.json`.
- **Feature** là Morgan count fingerprint `radius=2, nBits=1024` sinh từ
  `mudi_raw/id2smiles.pt`, cùng công thức với các dataset khác.

### Đánh giá có hướng

MUDI lưu mỗi cặp thuốc theo **cả hai chiều**, và file val/test xếp thành
`[nửa đầu = chiều xuôi | nửa sau = chiều ngược]`. `metric.py` so dòng `i` với dòng
`i + N/2` để tính exact-match mức 3 / mức 4 / vô hướng (option 1 / 2 / 3), nên:

- registry đặt `directed_eval: True` + `label_mapping` cho mudi, `trainer.py` gọi
  `directed_metrics()` thay cho báo cáo accuracy / macro-F1 / kappa;
- **thứ tự batch khi eval phải nguyên vẹn** — loader eval không shuffle và không
  `drop_last` nữa (trước đây `drop_last=True` cho mọi split, làm mất dòng cuối
  của cả test);
- chọn model / early stopping theo macro-F1 của option 1.

Đừng sắp xếp lại hay lọc bớt dòng trong các file split của MUDI: mất cấu trúc
nửa/nửa là metric sai mà không báo lỗi. `prepare_mudi.py` kiểm tra tính chất này
và từ chối ghi nếu không đạt.

## MecDDI

`data/initial/mecddi/` phủ đủ **1567/1567** drug.

- `DB_molecular_feats.pkl` đánh index theo cột **`Node_ID`**, *không* theo thứ tự
  dòng (khác DrugBank). Registry khai báo `feat_id_key='Node_ID'` nên code tự sắp
  lại theo drug id. **Code cũ lấy theo thứ tự dòng → mọi kết quả MecDDI chạy
  trước đây đều dùng fingerprint của thuốc khác; cần chạy lại.**
- `node2drugbank.json` giữ ánh xạ `node id -> DrugBank ID` cho cả 1567 id.
- Không có `relations_2hop.txt` → Decagon / TIGER không chạy được.
- Trong 1472 drug gốc có 238 dòng mà `Morgan_Features` không khớp với `SMILES`
  của chính nó — hai cột đến từ hai bản DrugBank khác nhau (không phải lệch
  dòng). Muốn đồng nhất một nguồn thì chạy `build_features.py --mode rebuild`,
  nhưng nó đổi feature của ~21% drug nên sẽ đổi luôn kết quả benchmark.

### Sinh lại feature từ CSV `Drugbank_ID,SMILES`

```
python build_features.py --dataset mecddi --csv ../../drugbank_smiles_map.csv          # dry run
python build_features.py --dataset mecddi --csv ../../drugbank_smiles_map.csv --write  # ghi thật (+ .bak)
```

Fingerprint dùng `GetHashedMorganFingerprint(mol, radius=2, nBits=1024)` (count
vector) — công thức này tái tạo đúng từng bit feature có sẵn, nên feature mới
cùng hệ với feature cũ.

CSV không có node id. Script lấy id từ `node_map_file` khai báo trong registry
(`node2drugbank.json`); nếu dataset chưa có node map thì nó dò theo thứ tự
DrugBank ID tăng dần và bỏ qua các chỗ mơ hồ. Đừng dùng `--ambiguous guess`:
với MecDDI cách đoán đó gán sai 4/7 drug so với node map thật.

## Định dạng file split

- **multiclass** (`drugbank`, `mecddi`, `mudi`): mỗi dòng `head tail rel`
- **multilabel** (`twosides`): mỗi dòng `head tail r0,r1,...,rN positive_flag`

## `initial/<dataset>/` cần gì

| file                     | model dùng tới                     | bắt buộc?                    |
|--------------------------|------------------------------------|------------------------------|
| `DB_molecular_feats.pkl` | mọi model có `--use_feat 1`        | có (trừ MSTE)                |
| `id2smiles.json`         | SSI-DDI, SA-DDI, SAGAN, TIGER      | chỉ khi chạy các model đó    |
| `relations_2hop.txt`     | Decagon, TIGER                     | chỉ khi chạy Decagon / TIGER |
| `node2drugbank.json`     | chỉ `build_features.py`            | không                        |

Dataset nào thiếu file sẽ báo lỗi rõ ràng ngay lúc load (`FileNotFoundError`
kèm đường dẫn mong đợi), thay vì chạy nhầm sang data của dataset khác. Riêng
drug thiếu SMILES thì SSI-DDI / SA-DDI thay bằng phân tử rỗng và in cảnh báo,
còn drug thiếu feature nhận vector 0 kèm cảnh báo.

## Thêm dataset mới

1. Bỏ data vào `data/<tên>_cluster/` (+ `_random/`) và `data/initial/<tên>/`.
2. Thêm một entry vào `DATASETS` trong `../dataset_registry.py`.

Không cần sửa gì trong `trainer.py`, `data_process.py`, `utils.py` hay `models/`
— code branch theo `args.task` (`multiclass` / `multilabel`), không theo tên dataset.

## Cache

TIGER ghi cache subgraph/molecule vào chính thư mục split
(`data/<dataset>_<type>/tiger_mol_sp.json`, `data/<dataset>_<type>/rw/`),
nên cache của các dataset cũng tách bạch. Xoá thư mục cache nếu bạn thay data.
