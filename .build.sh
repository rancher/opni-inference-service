IMAGE_NAME=tybalex/opni-inference:devy6
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
