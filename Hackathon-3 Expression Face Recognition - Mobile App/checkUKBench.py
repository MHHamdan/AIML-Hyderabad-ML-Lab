import h5py


print("Extracting keypoionts and local invariant descriptors \n \n ")
db = h5py.File("/home/mhamdan/myProjects2020/CBIR_Image_Search_Engine/FeatureExtraction/output/features_UKBench.hdf5",mode='r')
print(list(db.keys()))
print(db['image_ids'].shape)
print(db['index'].shape)
print(db['features'].shape)

print('Extracting keypoints and local invariant descriptors \n \n')
imageID = db['image_ids'][88]
print(imageID)

print('Extracting keypoints and local invariant descriptors \n \n')

(start, end) = db["index"][88]
print(start, end)
print(end - start)

print('Extracting keypoints and local invariant descriptors \n \n')
rows = db['features'][start:end]
print(rows.shape)
kps = rows[:, :2]
print(kps.shape)
descs = rows[:, 2:]
print(descs.shape)
