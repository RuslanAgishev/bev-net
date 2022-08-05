APP_NAME=ros:aanet
CONTAINER_NAME=aanet

all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build all images"
	# @echo "   1. make pull             - pull all images"
	@echo "   1. make clean              - remove all images"
	@echo ""

build:
	@docker build --tag ${APP_NAME} -f ./docker/Dockerfile ..

inference:
	@docker run \
			--runtime=nvidia \
		    -itd \
		    --rm \
		    --net=host \
		    -v $(shell pwd):/root/aanet \
		    -v /home/ruslan/datasets/kitti2012/:/data/KITTI/kitti_2012/ \
		    --name=${CONTAINER_NAME} \
		    ${APP_NAME} bash

exec: ## Run a bash in a running container
	@docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	@docker stop ${CONTAINER_NAME}

clean:
	@docker rmi -f ${APP_NAME}
