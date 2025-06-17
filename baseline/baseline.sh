python3 baseline.py --method gradcam --max_images 50  --image_dir datasets/imagenet --output_excel results/results_compare_2.xlsx
python3 baseline.py --method gradcamplusplus --max_images 50  --image_dir datasets/imagenet --output_excel results/results_compare_2.xlsx 
python3 baseline.py --method layercam --max_images 50  --image_dir datasets/imagenet --output_excel results/results_compare_2.xlsx 
python3 baseline.py --method scorecam --max_images 50  --image_dir datasets/imagenet --output_excel results/results_compare_2.xlsx 
python3 baseline.py --method ablationcam --max_images 50  --image_dir datasets/imagenet --output_excel results/results_compare_2.xlsx 
python3 baseline.py --method shapleycam --max_images 50  --image_dir datasets/imagenet --output_excel results/results_compare_2.xlsx 