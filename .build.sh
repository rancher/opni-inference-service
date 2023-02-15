IMAGE_NAME=tybalex/opni-inference:devd2
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
