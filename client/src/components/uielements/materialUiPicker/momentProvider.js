import React from 'react';
import { MuiPickersUtilsProvider } from '@material-ui/pickers';
import MomentUtils from '@date-io/moment';

export default props => (
  <MuiPickersUtilsProvider utils={MomentUtils} {...props} />
);
