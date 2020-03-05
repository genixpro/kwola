import styled, { keyframes } from 'styled-components';
import { palette } from 'styled-theme';
import WithDirection from '../../settings/withDirection';
import Buttons from '../../components/uielements/button';
import Icons from '../../components/uielements/icon';

const spinning = keyframes`
  from {
      transform:rotate(0deg);
  }
  to {
      transform:rotate(360deg);
  }
`;

const Icon = styled(Icons)``;

const Button = styled(Buttons)`
  width: 45px;
  height: 45px;
  min-width: auto;
  border-radius: 5px 0 0 5px;
  position: fixed;
  z-index: 1000;
  bottom: 100px;
  right: 0;
  padding: 0;
  transition: all 0.3s cubic-bezier(0.215, 0.61, 0.355, 1);

  &.active {
    right: 340px;

    @media only screen and (max-width: 767px) {
      right: 300px;
    }
  }

  ${Icon} {
    font-size: 18px;
    animation: ${spinning} 2.5s linear infinite;
  }
`;

const Main = styled.div`
  width: 100%;
  flex-grow: 1;
  background-color: ${palette('grey', 1)};
  padding: 0;
  margin-top: 64px;
  transition: all 0.2s ease-in-out;
  position: relative;

  &.fixedNav {
    width: calc(100% - 260px);
    margin-right: ${props => (props['data-rtl'] === 'rtl' ? 260 : 0)}px;
    margin-left: ${props => (props['data-rtl'] === 'rtl' ? 0 : 260)}px;

    .GridView {
      @media only screen and (max-width: 1140px) {
        width: 100%;
        margin: 0 0 20px;
      }

      @media only screen and (min-width: 1141px) and (max-width: 1320px) {
        width: calc(100% / 2 - 10px);
        margin: ${props =>
          props['data-rtl'] === 'rtl' ? '0 0 20px 20px' : '0 20px 20px 0'};

        &:nth-child(2n) {
          margin: ${props =>
            props['data-rtl'] === 'rtl' ? '0 0 20px 0' : '0 0 20px 0'};
        }

        &:nth-child(3n):not(:nth-child(2n)) {
          margin: ${props =>
            props['data-rtl'] === 'rtl' ? '0 0 20px 20px' : '0 20px 20px 0'};
        }
      }
    }

    .ListView {
      @media only screen and (max-width: 1130px) {
        flex-direction: column;

        .alGridImage {
          width: 100%;
          height: auto;
          overflow: hidden;
        }
      }
    }
  }

  @media only screen and (max-width: 767px) {
    flex-shrink: 0;
  }

  .scrollWrapper {
    height: calc(100vh - 65px);
  }

  .router-transition {
    position: relative;
  }

  .router-transition > div {
    position: absolute;
    width: 100%;
  }
`;

const Root = styled.div`
  width: 100%;
  z-index: 1;
  overflow: hidden;
`;

const AppFrame = styled.div`
  position: relative;
  display: flex;
  width: 100%;
  height: 100%;
`;

export default WithDirection(Main);
export { Root, AppFrame, Button, Icon };
