import styled from 'styled-components';
import CircularProgresses from '../../uielements/circularProgress';
import Icons from '../../uielements/icon';

const CircularProgress = styled(CircularProgresses)`
  margin-left: -25px;
`;

const Icon = styled(Icons)``;

const LoaderWrapper = styled.div`
  width: 100%;
  height: calc(100vh - 65px);
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.97);
  position: fixed;
  top: 0;
`;

const ErrorNotification = styled.div`
  width: 100%;
  height: calc(100vh - 65px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.97);
  position: fixed;
  top: 0;

  ${Icon} {
    font-size: 56px;
    color: #e0e0e0;
    margin-bottom: 20px;
  }

  h3 {
    font-size: 21px;
    color: #424242;
    font-weight: 500;
    margin: 0;
  }
`;

export { CircularProgress, ErrorNotification, Icon };
export default LoaderWrapper;
