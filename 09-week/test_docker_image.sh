
url="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
PORT=9000

echo "🧪 Invoking Lambda locally..."

curl --fail --show-error --silent \
  -X POST \
  "http://localhost:${PORT}/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
        "url": "'"${url}"'"
      }'

if [ $? -eq 0 ]; then
  echo -e "\n✅ Test passed!"
else
  echo -e "\n❌ Test failed!"
  exit 1
fi