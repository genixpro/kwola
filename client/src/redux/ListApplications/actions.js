const actions = {
  APPLICATION_LIST_REQUEST: 'APPLICATION_LIST_REQUEST',
  APPLICATION_LIST_SUCCESS_RESULT: 'APPLICATION_LIST_SUCCESS_RESULT',
  APPLICATION_LIST_ERROR_RESULT: 'APPLICATION_LIST_ERROR_RESULT',
  requestApplicationList: () => ({
    type: actions.APPLICATION_LIST_REQUEST
  }),
  applicationListSuccess: (applications) => ({
    type: actions.APPLICATION_LIST_SUCCESS_RESULT,
    applications
  }),
  applicationListError: () => ({
    type: actions.APPLICATION_LIST_ERROR_RESULT
  })
};
export default actions;
