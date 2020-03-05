const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

const as = values =>
  sleep(1000).then(() => {
    if (['foo@foo.com', 'bar@bar.com'].includes(values.email)) {
      const err = { email: 'Email already Exists' };
      throw err;
    }
  });
export default as;
