IMAGE_NAME=tybalex/opni-inference:devf
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
