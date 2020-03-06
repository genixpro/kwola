const actions = {
  APPLICATION_REQUEST: 'APPLICATION_REQUEST',
  APPLICATION_SUCCESS_RESULT: 'APPLICATION_SUCCESS_RESULT',
  APPLICATION_ERROR_RESULT: 'APPLICATION_ERROR_RESULT',

  NEW_TESTING_SEQUENCE_REQUEST: 'NEW_TESTING_SEQUENCE_REQUEST',
  NEW_TESTING_SEQUENCE_SUCCESS_RESULT: 'NEW_TESTING_SEQUENCE_SUCCESS_RESULT',
  NEW_TESTING_SEQUENCE_ERROR_RESULT: 'NEW_TESTING_SEQUENCE_ERROR_RESULT',

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
  }),


  requestNewTestingSequence: (applicationId) => ({
    type: actions.NEW_TESTING_SEQUENCE_REQUEST,
    applicationId: applicationId
  }),
  newTestingSequenceSuccess: (testingSequence) => ({
    type: actions.NEW_TESTING_SEQUENCE_SUCCESS_RESULT,
    testingSequence
  }),
  newTestingSequenceError: (error) => ({
    type: actions.NEW_TESTING_SEQUENCE_ERROR_RESULT,
    error: error
  }),

};
export default actions;
