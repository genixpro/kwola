import styled from 'styled-components';
import { palette } from 'styled-theme';
import Chips from '../../../components/uielements/chips';
import Icons from '../../../components/uielements/icon';

const Wrapper = styled.div`
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
`;

const Icon = styled(Icons)`
  font-size: 18px;
  color: ${palette('grey', 8)};
`;

const Chip = styled(Chips)`margin: 8px;`;

const ArrayChips = styled(Chips)`margin: 4px;`;

export { Chip, ArrayChips, Icon, Wrapper };
