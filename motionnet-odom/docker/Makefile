APP_NAME=ros:motionnet
CONTAINER_NAME=motionnet

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
	@docker build --tag ${APP_NAME} -f ./Dockerfile ..

inference:
	@docker run \
			--runtime=nvidia \
		    -itd \
		    --rm \
		    --net=host \
		    -v /home/ruslan/Desktop/DoEdu/src/motionnet:/root/motionnet \
		    --name=${CONTAINER_NAME} \
		    ${APP_NAME}

exec: ## Run a bash in a running container
	docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}

clean:
	@docker rmi -f ${APP_NAME}
