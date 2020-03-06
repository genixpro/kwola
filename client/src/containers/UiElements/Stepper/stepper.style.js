import styled from 'styled-components';
import { palette } from 'styled-theme';
import MobileSteppers from '../../../components/uielements/mobileStepper';
import Paper from '../../../components/uielements/paper';
import WithDirection from '../../../settings/withDirection';

const Root = styled.div`
  width: 100%;
`;

const ButtonContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
`;

const ButtonWrappers = styled.div`
  margin-right: ${props => (props['data-rtl'] === 'rtl' ? 'auto' : '8px')};
  margin-left: ${props => (props['data-rtl'] === 'rtl' ? '8px' : 'auto')};

  &:last-child {
    margin-right: ${props => (props['data-rtl'] === 'rtl' ? 'auto' : '0')};
    margin-left: ${props => (props['data-rtl'] === 'rtl' ? '0' : 'auto')};
  }
`;

const StepperContent = styled.div`
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 15px 0 30px;

  .instructions {
    margin: 8px 0 30px;
  }

  .completed {
    display: inline-block;
  }
`;

const MobileStepper = styled(MobileSteppers)`
  max-width: 400px;
  flex-grow: 1;
  display: flex;
  margin: auto;
`;

const MobileStepperTextWrapper = styled(Paper)`
  max-width: 384px;
  display: flex;
  align-items: center;
  height: 50px;
  margin: auto;
  padding-left: ${props => (props['data-rtl'] === 'rtl' ? 'auto' : '32px')};
  padding-right: ${props => (props['data-rtl'] === 'rtl' ? '32px' : 'auto')};
  margin-bottom: 20px;
  background: ${palette('grey', 0)};
`;

const ButtonWrapper = WithDirection(ButtonWrappers);
const MobileStepperText = WithDirection(MobileStepperTextWrapper);

export {
  Root,
  StepperContent,
  ButtonWrapper,
  MobileStepper,
  MobileStepperText,
  ButtonContainer,
};
