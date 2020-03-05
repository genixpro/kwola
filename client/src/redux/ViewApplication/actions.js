const actions = {
  APPLICATION_REQUEST: 'APPLICATION_REQUEST',
  APPLICATION_SUCCESS_RESULT: 'APPLICATION_SUCCESS_RESULT',
  APPLICATION_ERROR_RESULT: 'APPLICATION_ERROR_RESULT',
  requestApplication: (id) => ({
    type: actions.APPLICATION_REQUEST,
    _id: id
  }),
  applicationSuccess: (applications) => ({
    type: actions.APPLICATION_SUCCESS_RESULT,
    applications
  }),
  applicationError: () => ({
    type: actions.APPLICATION_ERROR_RESULT
  })
};
export default actions;
