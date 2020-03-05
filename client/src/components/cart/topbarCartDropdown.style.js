import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, boxShadow } from '../../settings/style-util';
import Icons from '../uielements/icon';
import WithDirection from '../../settings/withDirection';

const Icon = styled(Icons)`
  font-size: 14px;
  cursor: pointer;
  transition: color 0.3s;

  &:hover {
    color: ${palette('grey', 8)};
  }
`;

const TopbarCartWrapper = styled.div`
  width: auto;
  display: flex;
  align-items: center;
  padding: 20px 20px;
  margin-bottom: 10px;
  flex-shrink: 0;
  position: relative;
  background-color: #fff;
  ${boxShadow('0 0px 3px 0px rgba(0,0,0,0.2)')};
  ${transition()};

  &:last-child {
    margin-bottom: 0;
  }

  .itemImage {
    width: 50px;
    height: 50px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;

    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  .cartDetails {
    width: 100%;
    display: flex;
    padding: ${props =>
      props['data-rtl'] === 'rtl' ? '0 20px 0 0' : '0 0 0 20px'};
    flex-direction: column;
    text-align: ${props => (props['data-rtl'] === 'rtl' ? 'right' : 'left')};

    h3 {
      margin: 0 0 5px;
      line-height: 1;

      a {
        font-size: 13px;
        font-weight: 500;
        color: ${palette('grey', 8)};
        line-height: 1.3;
        text-decoration: none;
      }
    }

    p {
      display: flex;
      margin: 0;

      span {
        font-size: 12px;
        font-weight: 400;
        color: ${palette('grey', 6)};
        line-height: 1.2;

        &.itemMultiplier {
          padding: 0 3px;
        }
      }
    }
  }

  .itemRemove {
    position: absolute;
    right: ${props => (props['data-rtl'] === 'rtl' ? 'inherit' : '8px')};
    left: ${props => (props['data-rtl'] === 'rtl' ? '8px' : 'inherit')};
    font-size: 13px;
    font-weight: 500;
    top: 50%;
    margin-top: -10px;
    color: ${palette('text', 0)} !important;
    opacity: 0;
    ${transition()};
  }

  &:hover {
    .itemRemove {
      opacity: 1;
    }
  }
`;

export { Icon };
export default WithDirection(TopbarCartWrapper);
