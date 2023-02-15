IMAGE_NAME=tybalex/opni-inference:deve4
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
