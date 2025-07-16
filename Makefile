release:
	zip -r ture_claussen_tinyml_final_submission_v01.zip docs/ture_claussen_final_presentation.pdf project -x '*storage/*' -x '*.pio*' -x '*__pycache__*' -x '*data*' -x '*tts*' '*.git*' -x project/coeff_comparison.txt
