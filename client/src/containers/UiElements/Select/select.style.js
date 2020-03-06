import styled from 'styled-components';
import { FormControl } from '../../../components/uielements/form';
import Selects from '../../../components/uielements/select';
import { DialogContent as DialogContents } from '../../../components/uielements/dialogs';

const Form = styled.form`
  display: flex;
  flex-wrap: wrap;
`;

const Select = styled(Selects)`
  margin-top: 16px;
`;

const FormControls = styled(FormControl)`
  margin: 8px;
  min-width: calc(50% - 16px);
`;

const DialogContent = styled(DialogContents)`
  ${FormControls} {
    margin: 0;
    min-width: 100%;
  }
`;

export { Form, Select, FormControls, DialogContent };
