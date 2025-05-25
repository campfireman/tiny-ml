install-libs:
	cp -R .download/tflite-micro/tensorflow libs/tflite-micro
	cp -R .download/tflite-micro/third_party libs/tflite-micro

erase:
	pio run --target erase

clean:
	pio run --target clean

build:
	pio run

upload:
	pio run --target upload

serial:
	platformio device monitor