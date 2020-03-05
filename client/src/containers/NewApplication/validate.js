const validate = values => {
  const errors = {};
  const requiredFields = [
    'name',
    'url',
    'agreedTerms'
  ];
  requiredFields.forEach(field => {
    if (!values[field]) {
      errors[field] = 'Required';
    }
  });



  // if (
  //   values.email &&
  //   !/^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$/i.test(values.email)
  // ) {
  //   errors.email = 'Invalid email address';
  // }
  
  if (values.agreedTerms && values.agreedTerms !== true) {
    errors.agreedTerms = 'Please agree all statements';
  }
  return errors;
};
export default validate;
