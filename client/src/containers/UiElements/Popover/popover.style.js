import styled from 'styled-components';
import { palette } from 'styled-theme';
import Popover from '../../../components/uielements/popover';
import Button from '../../../components/uielements/button';

const StyledPopover = styled(Popover)`
  &.styledSupport {
    background-color: ${palette('primary', 0)};
  }
`;

const StyledButton = styled(Button)`
  &.hello {
    background-color: ${palette('primary', 0)};
  }
`;

export default StyledPopover;
export { StyledButton };
