# mudi_cluster — chưa có data

MUDI không được commit trong repo (trước đây copy đè lên thư mục drugbank lúc chạy).
Copy các file split vào đây: `train.txt`, `valid_S{0,1,2}.txt`, `test_S{0,1,2}.txt`,
mỗi dòng `head tail rel` với rel trong [0, 4):
0 = No Interaction, 1 = Synergism, 2 = Antagonism, 3 = New Effect.
