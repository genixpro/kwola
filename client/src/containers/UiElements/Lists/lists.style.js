import styled from 'styled-components';
import { palette } from 'styled-theme';
import Icons from '../../../components/uielements/icon';
import {
  FormGroup as FormGroups,
  FormControlLabel as FormControlLabels,
} from '../../../components/uielements/form/';

const Icon = styled(Icons)`
  font-size: 23px;
  color: ${palette('grey', 7)};
`;

const Root = styled.div`
  width: 100%;
  background: #fff;
  position: relative;
  overflow: auto;
  max-height: 300px;
  flex-grow: 1;
  max-width: 752px;
`;

const FormControlLabel = styled(FormControlLabels)``;

const FormGroup = styled(FormGroups)`padding: 30px;`;

export { Root, Icon, FormGroup, FormControlLabel };
