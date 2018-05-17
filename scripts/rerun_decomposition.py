from GFG_python.glm import FaceGenerator

fg = FaceGenerator(version='v1', save_dir='/Users/lukas/software/GFG_glm')
fg.run_decomposition(algorithm='pca', whiten=False)
