debug:
	# uv run python src/main.py debug=false +experiment=spiral_icrowd
	uv run python src/main.py -m debug=True \
	+experiment=tweet_eval_icrowd

spiral:
	uv run python src/main.py -m debug=false +experiment=spiral_learning_to_defer \
		trainer.loss.weight1=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
		trainer.loss.weight2=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
		trainer.loss.weight3=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
		trainer.seed=10,20,30,40,50 \
		name='spiral learning to defer with weight'

sst2:
	uv run python src/main.py -m debug=false +experiment=sst2_icrowd trainer.seed=10,20,30,40,50
	uv run python src/main.py -m debug=false +experiment=sst2_learning_to_defer \
		trainer.loss.weight1=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
		trainer.loss.weight2=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
		trainer.loss.weight3=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
		trainer.seed=10,20,30,40,50 \
		name='sst2 learning to defer with weight'

imdb:
	uv run python src/main.py -m debug=false +experiment=imdb_common_confusion,imdb_confusion,imdb_learning_to_defer,imdb_icrowd trainer.seed=10,20,30,40,50

run:
uv run python src/main.py -m +experiment=imdb_icrowd trainer.seed=10,20,30,40,50

classification:
	rye run python src/main.py -m '+experiment=glob(spiral*classification)' trainer.seed=10,20,30,40,50 name='spiral classification'

icrowd:
	uv run python src/main.py +experiment=imdb_icrowd_assignment name="imdb icrowd"
	uv run python src/main.py +experiment=tweet_eval_icrowd_assignment name='tweet eval icrowd'

train:
	rm -rf logs/stdout.txt
	qsub -g gcc50441 train.sh

test:
	rye run pytest -s

llm_annotation:
	# uv run python src/preprocess/llm_annotation.py -m '+experiment=glob(annotation_*)'
	uv run python src/preprocess/llm_annotation.py -m +experiment=annotation_poem_sentiment

docker:
	docker run -it  \
		--gpus all \
		-v ./:/task_assignment \
		test/moriyama #\
		#bash
		# -e LOCAL_UID=$(id -u $USER) \
		# -e LOCAL_GID=$(id -g $USER)\
		# -v /etc/group:/etc/group:ro \
		# -v /etc/passwd:/etc/passwd:ro \
