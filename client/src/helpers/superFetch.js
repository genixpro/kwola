const customHeader = () => ({
  'Content-Type': 'application/json',
  Accept: 'application/json'
});

const base = (method, url, data = {}) => {
  return fetch(url, {
    method,
    headers: customHeader(),
    body: JSON.stringify(data)
  })
    .then(response => response.json())
    .then(res => res)
    .catch(error => ({ error: 'Server Error' }));
};
const SuperFetch = {};
['get', 'post', 'put', 'delete'].forEach(method => {
  SuperFetch[method] = base.bind(null, method);
});
export default SuperFetch;
