from GFG_python.glm import FaceGenerator

fg = FaceGenerator(version='v1', save_dir='/Users/lukas/software/GFG_glm')
fg.load(h5_file='/Users/lukas/software/all_data.h5')
fg.fit_GLM(chunks=10)
fg.run_decomposition(algorithm='pca')
